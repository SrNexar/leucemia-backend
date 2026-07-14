# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="API Detección de Leucemia")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CARGAR MODELO ─────────────────────────────────────────────────────────────
print("Cargando modelo...")
modelo = tf.keras.models.load_model("Modelo/E1_custom_cnn_best.keras")
print("Modelo cargado correctamente.")

CLASES = ["ALL", "AML", "CLL", "CML", "Normal"]

DESCRIPCIONES = {
    "ALL": "Leucemia Linfoblástica Aguda — tipo más frecuente en población infantil.",
    "AML": "Leucemia Mielógena Aguda — proliferación anormal de células mieloides inmaduras.",
    "CLL": "Leucemia Linfocítica Crónica — acumulación de linfocitos maduros de apariencia uniforme.",
    "CML": "Leucemia Mieloide Crónica — células mieloides en distintos estadios de maduración.",
    "Normal": "Célula sanguínea sana — sin presencia de anomalías leucémicas."
}

# ── ENDPOINT PRINCIPAL ────────────────────────────────────────────────────────
@app.post("/predecir")
async def predecir(imagen: UploadFile = File(...)):
    contenido = await imagen.read()
    img = Image.open(io.BytesIO(contenido)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediccion = modelo.predict(img_array, verbose=0)
    clase_idx  = int(np.argmax(prediccion))
    clase      = CLASES[clase_idx]
    confianza  = float(prediccion[0][clase_idx]) * 100

    probabilidades = {
        CLASES[i]: round(float(prediccion[0][i]) * 100, 2)
        for i in range(len(CLASES))
    }

    # ── UMBRAL DE CONFIANZA ───────────────────────────────────────────
    UMBRAL = 70.0
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
# ── ENDPOINT DE SALUD ─────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"estado": "API funcionando correctamente"}

