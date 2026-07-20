"""
Script para precomputar estadísticas OOD (Out-of-Distribution).
Ejecutar UNA vez con el repositorio de imágenes de células sanguíneas.

Uso:
  python setup_ood.py --imagenes /ruta/a/carpeta/de/imagenes

Genera ood_stats.npz que main.py carga al iniciar.
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image


def build_feature_extractor(model):
    """Extrae features de la capa anterior a la salida softmax."""
    penultimate = model.layers[-2]
    return tf.keras.Model(inputs=model.inputs, outputs=penultimate.output)


def extract_features(img_array, feature_extractor):
    feats = feature_extractor.predict(img_array, verbose=0)
    return feats.flatten()


def mahalanobis_distance(x, mean, inv_cov):
    delta = x - mean
    return np.sqrt(np.dot(np.dot(delta, inv_cov), delta))


def main():
    parser = argparse.ArgumentParser(
        description="Precomputa estadísticas OOD para el detector de imágenes no-microscópicas"
    )
    parser.add_argument(
        "--imagenes", required=True,
        help="Directorio con imágenes de células sanguíneas (todas las clases)"
    )
    parser.add_argument(
        "--modelo", default="Modelo/E1_custom_cnn_best.keras",
        help="Ruta al modelo .keras"
    )
    parser.add_argument(
        "--salida", default="Modelo/ood_stats.npz",
        help="Archivo de salida para las estadísticas"
    )
    parser.add_argument(
        "--max", type=int, default=0,
        help="Máximo de imágenes a procesar (0 = todas)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.imagenes):
        print(f"Error: {args.imagenes} no es un directorio válido")
        sys.exit(1)

    print(f"Cargando modelo: {args.modelo}")
    model = tf.keras.models.load_model(args.modelo)
    feature_extractor = build_feature_extractor(model)
    print(f"Capa de features: {model.layers[-2].name} ({feature_extractor.output_shape[-1]} dimensiones)")

    print(f"\nBuscando imágenes recursivamente en: {args.imagenes}")
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    rutas = []
    for raiz, _, archivos in os.walk(args.imagenes):
        for fname in archivos:
            if any(fname.lower().endswith(e) for e in exts):
                rutas.append(os.path.join(raiz, fname))

    print(f"  Encontradas: {len(rutas)} imágenes en total")
    if not rutas:
        print("Error: no se encontraron imágenes.")
        sys.exit(1)

    if args.max and args.max < len(rutas):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(rutas), args.max, replace=False)
        rutas = [rutas[i] for i in sorted(indices)]
        print(f"  Muestreo: {args.max} imágenes seleccionadas aleatoriamente")

    features = []
    skipped = 0
    total = len(rutas)

    for i, path in enumerate(rutas):
        rel = os.path.relpath(path, args.imagenes)
        try:
            img = Image.open(path).convert("RGB").resize((224, 224))
            arr = np.array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            feat = extract_features(arr, feature_extractor)
            features.append(feat)
        except Exception as e:
            skipped += 1
            print(f"  [{i+1}/{total}] Omitida {rel}: {e}")
            continue

        if (i + 1) % 100 == 0 or i == total - 1:
            print(f"  [{i+1}/{total}] procesadas...")

    if len(features) < 10:
        print(f"Error: solo se procesaron {len(features)} imágenes. Se necesitan al menos 10.")
        sys.exit(1)

    features = np.array(features)
    print(f"\nProcesadas: {len(features)} imágenes ({skipped} omitidas)")
    print(f"Features shape: {features.shape}")

    mean = np.mean(features, axis=0)
    cov = np.cov(features.T)
    cov += np.eye(cov.shape[0]) * 1e-4
    inv_cov = np.linalg.inv(cov)

    distances = np.array([mahalanobis_distance(f, mean, inv_cov) for f in features])

    d_min = np.min(distances)
    d_max = np.max(distances)
    d_media = np.mean(distances)
    d_mediana = np.median(distances)
    p90 = np.percentile(distances, 90)
    p95 = np.percentile(distances, 95)
    p99 = np.percentile(distances, 99)

    # Umbral: la máxima distancia del conjunto de entrenamiento + 15% de margen.
    # Esto garantiza que NINGUNA imagen de células sanguíneas sea rechazada.
    threshold = d_max * 1.15

    print(f"\nEstadísticas de distancia Mahalanobis sobre {len(distances)} imágenes:")
    print(f"  Mínima:            {d_min:.2f}")
    print(f"  Media:             {d_media:.2f}")
    print(f"  Mediana:           {d_mediana:.2f}")
    print(f"  Percentil 90:      {p90:.2f}")
    print(f"  Percentil 95:      {p95:.2f}")
    print(f"  Percentil 99:      {p99:.2f}")
    print(f"  Máxima:            {d_max:.2f}")
    print(f"  Umbral OOD (max × 1.15): {threshold:.2f}")

    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
    np.savez(args.salida, mean=mean, inv_cov=inv_cov, threshold=threshold)
    print(f"\nEstadísticas guardadas en: {args.salida}")
    print("Listo.")


if __name__ == "__main__":
    main()
