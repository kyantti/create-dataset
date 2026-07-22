import os
import json
import numpy as np
from pathlib import Path

# Configuración
ANNOTATIONS_BASE = './outputs/segm-masks-2023-july-17-28'
OUTPUT_BASE = './outputs/cropped-hypercubes'

def analyze_coco_annotations():
    """Analiza los archivos COCO JSON y extrae estadísticas"""
    stats = {}
    total_images = 0
    total_detections = 0

    for class_id in range(4):  # C0 to C3
        class_name = f"C{class_id}"
        json_path = os.path.join(ANNOTATIONS_BASE, class_name, f"{class_name}_annotations_coco.json")

        if not os.path.exists(json_path):
            print(f"⚠️  {json_path} no encontrado")
            continue

        with open(json_path, 'r') as f:
            coco_data = json.load(f)

        n_images = len(coco_data['images'])
        n_annotations = len(coco_data['annotations'])

        stats[class_name] = {
            'images': n_images,
            'detections': n_annotations
        }

        total_images += n_images
        total_detections += n_annotations

        print(f"✅ {class_name}: {n_images} imágenes, {n_annotations} detecciones")

    stats['TOTAL'] = {
        'images': total_images,
        'detections': total_detections
    }

    return stats

def count_subcubes():
    """Cuenta los subcubos .npy generados por clase"""
    subcube_stats = {}
    total_subcubes = 0

    for class_id in range(4):
        class_name = f"C{class_id}"
        class_dir = os.path.join(OUTPUT_BASE, class_name)

        if not os.path.exists(class_dir):
            print(f"⚠️  Directorio {class_dir} no encontrado")
            subcube_stats[class_name] = 0
            continue

        npy_files = list(Path(class_dir).glob('*.npy'))
        n_subcubes = len(npy_files)
        subcube_stats[class_name] = n_subcubes
        total_subcubes += n_subcubes

        print(f"✅ {class_name}: {n_subcubes} subcubos extraídos")

    subcube_stats['TOTAL'] = total_subcubes
    return subcube_stats

def print_latex_table(coco_stats, subcube_stats):
    """Genera código LaTeX para la tabla de estadísticas"""
    print("\n" + "="*60)
    print("CÓDIGO LaTeX PARA LA TABLA:")
    print("="*60 + "\n")

    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Estadísticas del conjunto de datos generado por clase experimental.}")
    print(r"\label{tab:dataset_statistics}")
    print(r"\begin{tabular}{lccc}")
    print(r"\hline")
    print(r"\textbf{Clase} & \textbf{Imágenes RGB} & \textbf{Detecciones} & \textbf{Subcubos extraídos} \\")
    print(r"\hline")

    for class_id in range(4):
        class_name = f"C{class_id}"
        images = coco_stats.get(class_name, {}).get('images', 0)
        detections = coco_stats.get(class_name, {}).get('detections', 0)
        subcubes = subcube_stats.get(class_name, 0)
        print(f"{class_name} & {images} & {detections} & {subcubes} \\\\")

    print(r"\hline")
    total_images = coco_stats.get('TOTAL', {}).get('images', 0)
    total_detections = coco_stats.get('TOTAL', {}).get('detections', 0)
    total_subcubes = subcube_stats.get('TOTAL', 0)
    print(f"\\textbf{{Total}} & \\textbf{{{total_images}}} & \\textbf{{{total_detections}}} & \\textbf{{{total_subcubes}}} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")

def main():
    print("\n🔍 Analizando anotaciones COCO...")
    coco_stats = analyze_coco_annotations()

    print("\n🔍 Contando subcubos hiperespectrales...")
    subcube_stats = count_subcubes()

    print_latex_table(coco_stats, subcube_stats)

    print("\n" + "="*60)
    print("RESUMEN:")
    print("="*60)
    print(f"📊 Total imágenes procesadas: {coco_stats['TOTAL']['images']}")
    print(f"🎯 Total detecciones: {coco_stats['TOTAL']['detections']}")
    print(f"📦 Total subcubos extraídos: {subcube_stats['TOTAL']}")
    print(f"📏 Bandas espectrales: 448")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()