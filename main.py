from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image, ImageStat
import io

app = FastAPI(title="API Detección de Leucemia")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Cargando modelo...")
modelo = tf.keras.models.load_model("Modelo/E1_custom_cnn_best.keras")
print("Modelo cargado correctamente.")

CLASES = ["ALL", "AML", "CLL", "CML", "Normal"]

DESCRIPCIONES = {
    "ALL": "Leucemia Linfoblástica Aguda — tipo más frecuente en población pediátrica.",
    "AML": "Leucemia Mielógena Aguda — proliferación anormal de células mieloides inmaduras.",
    "CLL": "Leucemia Linfocítica Crónica — acumulación de linfocitos maduros de apariencia uniforme.",
    "CML": "Leucemia Mieloide Crónica — células mieloides en distintos estadios de maduración.",
    "Normal": "Célula sanguínea sana — sin presencia de anomalías leucémicas."
}

def es_imagen_microscopica(img: Image.Image) -> bool:
    """
    Verifica si la imagen tiene características visuales
    compatibles con microscopía de células sanguíneas.
    """
    img_rgb = img.convert("RGB")
    stat = ImageStat.Stat(img_rgb)

    medias = stat.mean      # media por canal R, G, B
    desvios = stat.stddev   # desviación estándar por canal

    r_mean, g_mean, b_mean = medias
    r_std, g_std, b_std = desvios

    # Imágenes de microscopía con Giemsa tienen:
    # - Fondo claro (valores altos de R y G)
    # - Tonos púrpura/azul en las células (B relativamente alto)
    # - Bajo contraste global (desviación estándar moderada)

    # Rechazar imágenes muy saturadas o con colores muy naturales
    # como fotografías de personas, paisajes, etc.

    # Condición 1: el canal rojo no debe ser extremadamente dominante
    # (fotos de personas tienen R muy alto)
    if r_mean > 200 and r_mean > b_mean * 1.5:
        return False

    # Condición 2: la imagen debe tener suficiente variación tonal
    # (imágenes de microscopía tienen células oscuras sobre fondo claro)
    variacion_total = r_std + g_std + b_std
    if variacion_total < 30:
        return False

    # Condición 3: no debe ser una imagen completamente saturada
    # o de colores muy vivos (fotografías naturales)
    if r_mean > 180 and g_mean > 150 and b_mean > 130:
        # Parece una foto natural con colores cálidos
        diferencia_rg = abs(r_mean - g_mean)
        diferencia_rb = abs(r_mean - b_mean)
        if diferencia_rg < 30 and diferencia_rb < 50:
            return False

    return True


@app.post("/predecir")
async def predecir(imagen: UploadFile = File(...)):
    contenido = await imagen.read()
    img = Image.open(io.BytesIO(contenido)).convert("RGB")

    # ── VALIDACIÓN PREVIA ─────────────────────────────────────────────
    if not es_imagen_microscopica(img):
        return {
            "clase":          "Indeterminado",
            "confianza":      0.0,
            "descripcion":    "La imagen no corresponde a una muestra de microscopía de células sanguíneas. Por favor utilice imágenes obtenidas mediante microscopía óptica con tinción de Giemsa.",
            "probabilidades": {c: 0.0 for c in CLASES},
            "advertencia":    True
        }
    # ─────────────────────────────────────────────────────────────────

    img_redim  = img.resize((224, 224))
    img_array  = np.array(img_redim) / 255.0
    img_array  = np.expand_dims(img_array, axis=0)

    prediccion = modelo.predict(img_array, verbose=0)
    clase_idx  = int(np.argmax(prediccion))
    clase      = CLASES[clase_idx]
    confianza  = float(prediccion[0][clase_idx]) * 100

    probabilidades = {
        CLASES[i]: round(float(prediccion[0][i]) * 100, 2)
        for i in range(len(CLASES))
    }

    # ── UMBRAL DE CONFIANZA ───────────────────────────────────────────
    UMBRAL = 85.0
    if confianza < UMBRAL:
        return {
            "clase":          "Indeterminado",
            "confianza":      round(confianza, 2),
            "descripcion":    "La imagen no presenta características suficientemente claras para realizar una clasificación confiable. Se recomienda utilizar una imagen de microscopía con tinción de Giemsa adecuada.",
            "probabilidades": probabilidades,
            "advertencia":    True
        }
    # ─────────────────────────────────────────────────────────────────

    return {
        "clase":          clase,
        "confianza":      round(confianza, 2),
        "descripcion":    DESCRIPCIONES[clase],
        "probabilidades": probabilidades,
        "advertencia":    False
    }


@app.get("/")
def health():
    return {"estado": "API funcionando correctamente"}