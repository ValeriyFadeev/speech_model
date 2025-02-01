# app.py
import os
import tempfile
import torch
import whisper
import spacy
from pyannote.audio import Pipeline
from pydub import AudioSegment

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# Глобальные переменные для моделей
asr_model = None
diar_pipeline = None
nlp = None

device = "cuda" if torch.cuda.is_available() else "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Вместо @app.on_event("startup") используем lifespan-событие.
    Здесь загружаем модели при старте и (опционально) можем освободить ресурсы при завершении.
    """
    global asr_model, diar_pipeline, nlp

    # 1) Whisper
    asr_model = whisper.load_model("base", device=device)
    # 2) pyannote.audio (если есть HF_TOKEN — используем)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        diar_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",use_auth_token=hf_token)
    else:
        diar_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

    # 3) spaCy
    nlp = spacy.load("ru_core_news_sm")

    # Код внутри yield — это "runtime" приложения
    yield

    # Здесь можно добавить логику остановки/освобождения ресурсов, если нужно
    # (Например, diar_pipeline и т.п.)

app = FastAPI(lifespan=lifespan)

def convert_to_wav(input_file_path, sr=16000):
    audio = AudioSegment.from_file(input_file_path)
    audio = audio.set_channels(1).set_frame_rate(sr)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_name = tmp.name
    audio.export(tmp_name, format="wav")
    return tmp_name

def assign_names_to_speakers(segments, matched_speakers, nlp):
    """
    Сопоставляет (по возможности) имена спикерам через spaCy NER (тип "PER").
    Если имя не найдено — Speaker_X.
    """
    speaker_to_name = {}
    for seg, spk_label in zip(segments, matched_speakers):
        text = seg["text"]
        # Пропускаем, если уже есть имя у этого спикера
        if spk_label in speaker_to_name:
            continue

        doc = nlp(text)
        person_names = [ent.text for ent in doc.ents if ent.label_ == "PER"]
        if person_names:
            speaker_to_name[spk_label] = person_names[0]
        else:
            speaker_to_name[spk_label] = None

    unnamed_count = 1
    for spk_label in set(matched_speakers):
        if speaker_to_name.get(spk_label) is None:
            speaker_to_name[spk_label] = f"Speaker_{unnamed_count}"
            unnamed_count += 1

    return speaker_to_name

def process_audio(input_file_path):
    wav_path = convert_to_wav(input_file_path, sr=16000)
    try:
        # Шаг 1: Whisper
        stt_result = asr_model.transcribe(wav_path, language='ru')
        segments = stt_result.get("segments", [])  # [{start, end, text}, ...]

        # Шаг 2: pyannote диаризация
        diarization = diar_pipeline(wav_path)
        diar_results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diar_results.append((turn.start, turn.end, speaker))
        diar_results.sort(key=lambda x: x[0])

        # Шаг 3: сопоставляем сегменты Whisper с временными интервалами диаризации
        matched_speakers = []
        for seg in segments:
            seg_start = seg["start"]
            spk_label = "UNK"
            for (dst, den, spk) in diar_results:
                if dst <= seg_start < den:
                    spk_label = spk
                    break
            matched_speakers.append(spk_label)

        # Шаг 4: присвоение имён (spaCy NER)
        speaker_name_map = assign_names_to_speakers(segments, matched_speakers, nlp)

        # Шаг 5: формируем итоговый текст
        output_lines = []
        for seg, spk_label in zip(segments, matched_speakers):
            assigned_name = speaker_name_map.get(spk_label, f"Speaker_{spk_label}")
            text = seg["text"]
            output_lines.append(f"{assigned_name}: {text}")

        return output_lines

    finally:
        # Удаляем временный WAV
        os.unlink(wav_path)

@app.post("/transcribe")
def transcribe_file(file: UploadFile = File(...)):
    """
    Принимает аудиофайл (mp3, mp4, wav и т.д.) через form-data (ключ 'file').
    Возвращает JSON с расшифровкой (массив строк).
    """
    if not file or not file.filename:
        return JSONResponse(content={"error": "No file uploaded"}, status_code=400)

    # Сохраняем загруженный файл во временное место
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_name = tmp.name
        tmp.write(file.file.read())

    # Обрабатываем файл
    lines = process_audio(tmp_name)

    # Удаляем временный входной файл
    os.unlink(tmp_name)

    # Возвращаем массив расшифрованных строк
    return {"transcription": lines}
