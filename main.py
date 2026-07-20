from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image, ImageStat
import io
import os

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


# ═══════════════════════════════════════════════════════════════════════
# DETECTOR OOD POR DISTANCIA DE MAHALANOBIS
# ═══════════════════════════════════════════════════════════════════════

class OODDetector:
    """Detector de imágenes fuera de distribución usando distancia de
    Mahalanobis sobre el espacio de features de la penúltima capa del modelo."""

    def __init__(self, model, stats_path="Modelo/ood_stats.npz"):
        self.feature_extractor = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.layers[-2].output
        )
        self.mean = None
        self.inv_cov = None
        self.threshold = None
        if os.path.exists(stats_path):
            self._cargar(stats_path)
        else:
            print(f"  ⚠ ood_stats.npz no encontrado en {stats_path}")
            print(f"    Ejecutá: python setup_ood.py --imagenes /ruta/a/tus/imagenes")

    def _cargar(self, path):
        data = np.load(path)
        self.mean = data["mean"]
        self.inv_cov = data["inv_cov"]
        self.threshold = float(data["threshold"])
        print(f"  ✓ Detector OOD cargado ({len(self.mean)} dims, umbral={self.threshold:.2f})")

    @property
    def activo(self) -> bool:
        return self.mean is not None

    def es_ood(self, img_array: np.ndarray) -> bool:
        """True si la imagen está fuera de distribución (no es célula sanguínea)."""
        if not self.activo:
            return False
        feats = self.feature_extractor.predict(img_array, verbose=0).flatten()
        delta = feats - self.mean
        dist = float(np.sqrt(np.dot(np.dot(delta, self.inv_cov), delta)))
        return dist > self.threshold


ood_detector = OODDetector(modelo)


# ═══════════════════════════════════════════════════════════════════════
# FILTRO HEURÍSTICO MEJORADO
# ═══════════════════════════════════════════════════════════════════════

def _porcentaje_pixeles_claros(arr: np.ndarray, umbral=200) -> float:
    """Porcentaje de píxeles con valor > umbral en los 3 canales (fondo claro)."""
    claros = np.all(arr > umbral, axis=-1)
    return np.mean(claros) * 100


def _porcentaje_pixeles_oscuros(arr: np.ndarray, umbral=80) -> float:
    """Porcentaje de píxeles con valor < umbral en los 3 canales (núcleos teñidos)."""
    oscuros = np.all(arr < umbral, axis=-1)
    return np.mean(oscuros) * 100


def _razon_azul_rojo(medias) -> float:
    """Razón entre canal azul y rojo (Giemsa tiñe núcleos de azul/púrpura)."""
    r, g, b = medias
    return b / max(r, 1)


def _tiene_tono_piel(arr: np.ndarray) -> bool:
    """Detecta si la imagen contiene predominantemente tonos de piel humana.
    Piel: R>G>B con R en rango medio-alto, diferencia R-B moderada."""
    r = arr[:, :, 0].astype(float)
    g = arr[:, :, 1].astype(float)
    b = arr[:, :, 2].astype(float)

    mascara_piel = (
        (r > g) & (g > b) &
        (r > 100) & (r < 240) &
        (g > 60) & (g < 210) &
        ((r - b) < 100) &
        ((r - g) < 70)
    )
    proporcion = np.mean(mascara_piel)
    return proporcion > 0.35


def _tiene_verde_dominante(medias) -> bool:
    """Detecta vegetación/escenas naturales: verde muy por encima del azul."""
    r, g, b = medias
    return (g > b * 1.25) and (g > r * 0.9)


def _entropia_std_local(arr: np.ndarray, tam_bloque=16) -> float:
    """Desviación estándar de la media local por bloques.
    Imágenes de microscopía tienen fondo uniforme con núcleos aislados
    → desviación local moderada. Fotografías tienen variación continua → baja/alta."""
    h, w = arr.shape[:2]
    medias_locales = []
    for y in range(0, h - tam_bloque, tam_bloque):
        for x in range(0, w - tam_bloque, tam_bloque):
            bloque = arr[y:y+tam_bloque, x:x+tam_bloque, :]
            medias_locales.append(np.mean(bloque))
    return float(np.std(medias_locales))


def es_imagen_microscopica(img: Image.Image) -> bool:
    """
    Verifica que la imagen tenga características visuales compatibles
    con microscopía óptica de células sanguíneas con tinción de Giemsa.

    Devuelve False para: fotos de personas, animales, paisajes, objetos cotidianos.
    """
    img_rgb = img.convert("RGB")
    stat = ImageStat.Stat(img_rgb)
    medias = stat.mean
    desvios = stat.stddev

    r_mean, g_mean, b_mean = medias
    r_std, g_std, b_std = desvios

    # ── 1. Fondo claro dominante ────────────────────────────────────
    # Microscopía: >40% de la imagen es fondo claro (campo brillante)
    arr = np.array(img_rgb.resize((112, 112)))
    pct_claro = _porcentaje_pixeles_claros(arr, umbral=180)
    if pct_claro < 25:
        return False

    # ── 2. Presencia de núcleos teñidos oscuros ──────────────────────
    # Debe haber al menos algunos píxeles oscuros (núcleos celulares)
    pct_oscuro = _porcentaje_pixeles_oscuros(arr, umbral=70)
    if pct_oscuro < 0.5:
        return False

    # ── 3. Balance azul/rojo compatible con tinción ──────────────────
    # Giemsa: núcleos púrpura-azulados (B elevado), citoplasma rosa (R moderado)
    razon_b_r = _razon_azul_rojo(medias)
    if razon_b_r < 0.65:
        return False

    # ── 4. Rechazo de tonos de piel ──────────────────────────────────
    if _tiene_tono_piel(arr):
        return False

    # ── 5. Rechazo de vegetación / paisajes ──────────────────────────
    if _tiene_verde_dominante(medias):
        return False

    # ── 6. Variación cromática suficiente ────────────────────────────
    variacion_total = r_std + g_std + b_std
    if variacion_total < 20:
        return False

    # ── 7. Saturación extrema (fotos muy coloridas) ──────────────────
    if r_mean > 190 and g_mean > 170 and b_mean > 150:
        return False

    return True


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

RESPUESTA_RECHAZO = {
    "clase": "Indeterminado",
    "confianza": 0.0,
    "descripcion": (
        "La imagen no corresponde a una muestra de microscopía de células "
        "sanguíneas. Utilice imágenes obtenidas mediante microscopía óptica "
        "con tinción de Giemsa."
    ),
    "probabilidades": {c: 0.0 for c in CLASES},
    "advertencia": True
}

RESPUESTA_BAJA_CONFIANZA_TEMPLATE = (
    "La imagen no presenta características suficientemente claras para una "
    "clasificación confiable. Se recomienda utilizar una imagen de microscopía "
    "con tinción de Giemsa adecuada."
)


@app.post("/predecir")
async def predecir(imagen: UploadFile = File(...)):
    contenido = await imagen.read()
    img = Image.open(io.BytesIO(contenido)).convert("RGB")

    # ── FILTRO 1: heurísticas visuales ──────────────────────────────
    if not es_imagen_microscopica(img):
        return RESPUESTA_RECHAZO

    # ── Preprocesamiento ────────────────────────────────────────────
    img_redim = img.resize((224, 224))
    img_array = np.array(img_redim) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ── FILTRO 2: distancia de Mahalanobis sobre features del modelo ─
    if ood_detector.activo and ood_detector.es_ood(img_array):
        return RESPUESTA_RECHAZO

    # ── Predicción ──────────────────────────────────────────────────
    prediccion = modelo.predict(img_array, verbose=0)
    clase_idx = int(np.argmax(prediccion))
    clase = CLASES[clase_idx]
    confianza = float(prediccion[0][clase_idx]) * 100

    probabilidades = {
        CLASES[i]: round(float(prediccion[0][i]) * 100, 2)
        for i in range(len(CLASES))
    }

    # ── FILTRO 3: umbral de confianza ───────────────────────────────
    UMBRAL = 85.0
    if confianza < UMBRAL:
        return {
            "clase": "Indeterminado",
            "confianza": round(confianza, 2),
            "descripcion": RESPUESTA_BAJA_CONFIANZA_TEMPLATE,
            "probabilidades": probabilidades,
            "advertencia": True
        }

    return {
        "clase": clase,
        "confianza": round(confianza, 2),
        "descripcion": DESCRIPCIONES[clase],
        "probabilidades": probabilidades,
        "advertencia": False
    }


@app.get("/")
def health():
    return {"estado": "API funcionando correctamente"}
