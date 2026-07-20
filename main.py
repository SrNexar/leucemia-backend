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
        self.threshold_raw = float(data["threshold"])
        factor = float(os.environ.get("OOD_MULTIPLIER", 3))
        self.threshold = self.threshold_raw * factor
        print(f"  ✓ Detector OOD cargado ({len(self.mean)} dims)")
        print(f"    Umbral base: {self.threshold_raw:.2f} × {factor} = {self.threshold:.2f}")

    @property
    def activo(self) -> bool:
        return self.mean is not None

    def distancia(self, img_array: np.ndarray) -> float:
        """Distancia de Mahalanobis. A menor distancia, más típica es la imagen."""
        if not self.activo:
            return 0.0
        feats = self.feature_extractor.predict(img_array, verbose=0).flatten()
        delta = feats - self.mean
        return float(np.sqrt(np.dot(np.dot(delta, self.inv_cov), delta)))

    def es_ood(self, img_array: np.ndarray) -> bool:
        return self.distancia(img_array) > self.threshold


ood_detector = OODDetector(modelo)


# ═══════════════════════════════════════════════════════════════════════
# FILTRO HEURÍSTICO — solo rechaza lo OBVIAMENTE no-microscópico
# ═══════════════════════════════════════════════════════════════════════

def _tiene_tono_piel(arr: np.ndarray) -> bool:
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
    return np.mean(mascara_piel) > 0.50


def _tiene_verde_dominante(medias) -> bool:
    r, g, b = medias
    return (g > b * 1.3) and (g > r * 0.95)


def es_imagen_microscopica(img: Image.Image):
    """
    Retorna (ok: bool, motivo: str).
    Solo rechaza imágenes que claramente NO son microscopía:
    fotos de personas, paisajes/vegetación, imágenes extremadamente
    planas o saturadas.
    """
    img_rgb = img.convert("RGB")
    stat = ImageStat.Stat(img_rgb)
    medias = stat.mean
    desvios = stat.stddev
    r_mean, g_mean, b_mean = medias
    r_std, g_std, b_std = desvios

    arr = np.array(img_rgb)

    # ── 1. Imagen totalmente plana (folder vacío, ruido uniforme) ───
    variacion_total = r_std + g_std + b_std
    if variacion_total < 15:
        return False, "imagen sin variación cromática suficiente"

    # ── 2. Foto de persona (tonos de piel dominantes) ────────────────
    if _tiene_tono_piel(arr):
        return False, "posible fotografía de persona (tonos de piel dominantes)"

    # ── 3. Paisaje / naturaleza (verde dominante) ────────────────────
    if _tiene_verde_dominante(medias):
        return False, "posible paisaje o vegetación (canal verde dominante)"

    # ── 4. Imagen extremadamente brillante y colorida (publicidad, IA) ─
    if r_mean > 200 and g_mean > 185 and b_mean > 170:
        return False, "imagen excesivamente brillante/saturada"

    return True, ""


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

def _rechazo(motivo: str, distancia: float = 0.0) -> dict:
    return {
        "clase": "Indeterminado",
        "confianza": 0.0,
        "descripcion": f"La imagen no corresponde a una muestra de microscopía de células sanguíneas. Motivo: {motivo}.",
        "probabilidades": {c: 0.0 for c in CLASES},
        "advertencia": True,
        "motivo": motivo,
        "distancia_ood": round(distancia, 2),
    }


@app.post("/predecir")
async def predecir(imagen: UploadFile = File(...)):
    contenido = await imagen.read()

    if not contenido:
        return _rechazo("archivo vacío")

    try:
        img = Image.open(io.BytesIO(contenido)).convert("RGB")
    except Exception:
        return _rechazo("el archivo no es una imagen válida (formatos aceptados: PNG, JPG, BMP, TIFF)")

    # ── FILTRO 1: heurísticas visuales básicas ───────────────────────
    ok, motivo_heur = es_imagen_microscopica(img)
    if not ok:
        return _rechazo(motivo_heur)

    # ── Preprocesamiento ────────────────────────────────────────────
    img_redim = img.resize((224, 224))
    img_array = np.array(img_redim) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ── FILTRO 2: distancia de Mahalanobis ──────────────────────────
    dist = ood_detector.distancia(img_array)
    if ood_detector.activo and ood_detector.es_ood(img_array):
        return _rechazo(
            f"la imagen no coincide con la distribución de células sanguíneas "
            f"(distancia Mahalanobis {dist:.1f}, umbral {ood_detector.threshold:.1f})",
            distancia=dist
        )

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
    UMBRAL = 70.0
    if confianza < UMBRAL:
        return {
            "clase": "Indeterminado",
            "confianza": round(confianza, 2),
            "descripcion": (
                "La imagen no presenta características suficientemente claras "
                "para una clasificación confiable. Se recomienda utilizar una "
                "imagen de microscopía con tinción de Giemsa adecuada."
            ),
            "probabilidades": probabilidades,
            "advertencia": True,
            "motivo": f"confianza {confianza:.1f}% por debajo del umbral {UMBRAL}%",
            "distancia_ood": round(dist, 2),
        }

    return {
        "clase": clase,
        "confianza": round(confianza, 2),
        "descripcion": DESCRIPCIONES[clase],
        "probabilidades": probabilidades,
        "advertencia": False,
        "distancia_ood": round(dist, 2),
    }


@app.get("/")
def health():
    return {"estado": "API funcionando correctamente"}
