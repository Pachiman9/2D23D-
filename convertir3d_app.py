import os
import cv2
import torch
import numpy as np
import PySimpleGUI as sg
from tqdm import tqdm

# --- Funciones de IA ---
def load_midas_model(model_path=None):
    try:
        if model_path and os.path.exists(model_path):
            model = torch.jit.load(model_path)
        else:
            model_type = "DPT_Hybrid"
            model = torch.hub.load("intel-isl/MiDaS", model_type)
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        return model
    except Exception as e:
        sg.popup_error(f"No se pudo cargar el modelo MiDaS:\n{e}")
        exit()

def predict_depth(model, frame):
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    input_batch = transform(frame).unsqueeze(0)
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    return depth_map

def depth_to_stereo(frame, depth_map, shift):
    h, w, c = frame.shape
    left = np.zeros_like(frame)
    right = np.zeros_like(frame)
    for y in range(h):
        for x in range(w):
            dx = int(depth_map[y, x] * shift)
            left[y, min(w-1, x-dx)] = frame[y, x]
            right[y, min(w-1, x+dx)] = frame[y, x]
    stereo = np.concatenate((left, right), axis=1)
    return stereo

# --- Interfaz gráfica ---
sg.theme('DarkBlue3')
layout = [
    [sg.Text("Selecciona vídeo de entrada:"), sg.InputText(key="-IN-"), sg.FileBrowse(file_types=(("Vídeos", "*.mp4 *.avi *.mov"),))],
    [sg.Text("Selecciona archivo de salida:"), sg.InputText(key="-OUT-"), sg.FileSaveAs(file_types=(("Vídeos", "*.mp4"),))],
    [sg.Text("Intensidad 3D:"), sg.Slider(range=(5,30), orientation='h', size=(34,20), default_value=15, key="-SHIFT-")],
    [sg.Text("Resolución (% de original):"), sg.Slider(range=(25,100), orientation='h', size=(34,20), default_value=100, key="-RES-")],
    [sg.Checkbox("Previsualizar primeros 3 segundos", key="-PREVIEW-", default=True)],
    [sg.Button("Convertir"), sg.Button("Salir")],
    [sg.ProgressBar(100, orientation='h', size=(50,20), key='-PROG-')]
]

window = sg.Window("Convertidor 2D a 3D", layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event=="Salir":
        break
    if event == "Convertir":
        input_path = values["-IN-"]
        output_path = values["-OUT-"]
        shift = int(values["-SHIFT-"])
        res_percent = int(values["-RES-"])
        preview = values["-PREVIEW-"]

        if not os.path.exists(input_path):
            sg.popup_error("Archivo de entrada no encontrado.")
            continue
        if not output_path:
            sg.popup_error("Debe seleccionar un archivo de salida.")
            continue

        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                sg.popup_error("No se pudo abrir el vídeo.")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * res_percent / 100)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * res_percent / 100)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (w*2, h))

            model = load_midas_model()  # Por defecto descarga MiDaS la primera vez

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            preview_frames = int(min(frame_count, fps*3)) if preview else 0

            for i in tqdm(range(frame_count), desc="Procesando frames"):
                ret, frame = cap.read()
                if not ret:
                    break
                if res_percent != 100:
                    frame = cv2.resize(frame, (w, h))
                depth_map = predict_depth(model, frame)
                stereo_frame = depth_to_stereo(frame, depth_map, shift)
                out.write(stereo_frame)
                prog = int((i+1)/frame_count*100)
                window['-PROG-'].update(prog)
                if preview and i+1 >= preview_frames:
                    break  # Solo previsualización

            cap.release()
            out.release()
            sg.popup("¡Conversión completada!", f"Vídeo 3D guardado en:\n{output_path}")

        except Exception as e:
            sg.popup_error(f"Ocurrió un error durante la conversión:\n{e}")

window.close()
