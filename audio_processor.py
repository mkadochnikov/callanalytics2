#!/usr/bin/env python3
"""
Класс для обработки аудио и транскрибации
"""

from config import *


class AudioProcessor:
    """Класс для обработки аудиофайлов и транскрибации"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.audio_dir = base_dir / "audio"
        self.transcripts_dir = base_dir / "transcripts"
        self.whisper_model = None
        self._init_whisper()

    def _init_whisper(self):
        """Инициализация модели Whisper"""
        try:
            logger.info(f"Загрузка модели Whisper ({WHISPER_MODEL_SIZE})...")
            self.whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
            logger.info("✅ Модель Whisper загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки Whisper: {e}")
            self.whisper_model = None

    def download_audio_file(self, file_info: Dict, filename: str, date_str: str, session: requests.Session) -> Optional[
        str]:
        """Скачивает аудиофайл"""
        try:
            # Создаем папку для даты
            date_audio_dir = self.audio_dir / date_str
            date_audio_dir.mkdir(exist_ok=True)

            file_path = date_audio_dir / f"{filename}.mp3"

            if file_path.exists():
                logger.info(f"Файл уже существует: {file_path}")
                return str(file_path)

            file_url = file_info.get('url', '')
            if not file_url:
                return None

            response = session.get(file_url, allow_redirects=True, timeout=60)

            if response.status_code == 200:
                content = response.content

                if content[:3] == b'ID3' or b'Lavf' in content[:100] or len(content) > 1000:
                    with open(file_path, 'wb') as f:
                        f.write(content)

                    logger.info(f"Скачано: {file_path}")
                    return str(file_path)

            return None

        except Exception as e:
            logger.error(f"Ошибка скачивания файла: {e}")
            return None

    def transcribe_audio(self, audio_path: str) -> str:
        """Транскрибирует аудиофайл в текст с помощью Whisper"""
        try:
            if not self.whisper_model:
                return "Ошибка: модель Whisper не загружена"

            logger.info(f"Транскрибация: {audio_path}")

            # Транскрибируем с помощью Whisper
            result = self.whisper_model.transcribe(
                audio_path,
                language="ru",  # Указываем русский язык
                task="transcribe"
            )

            transcript_text = result["text"].strip()

            if transcript_text:
                logger.info(f"Транскрибация завершена: {len(transcript_text)} символов")
                return transcript_text
            else:
                return "Не удалось распознать речь"

        except Exception as e:
            logger.error(f"Ошибка транскрибации {audio_path}: {e}")
            return "Ошибка транскрибации"

    def save_transcript(self, transcript: str, filename: str, date_str: str):
        """Сохраняет транскрипцию в файл"""
        transcripts_date_dir = self.transcripts_dir / date_str
        transcripts_date_dir.mkdir(exist_ok=True)

        transcript_file = transcripts_date_dir / f"{filename}.txt"

        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(transcript)

    def load_transcript(self, filename: str, date_str: str) -> Optional[str]:
        """Загружает транскрипцию из файла"""
        transcript_file = self.transcripts_dir / date_str / f"{filename}.txt"

        if transcript_file.exists():
            try:
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Ошибка загрузки транскрипции: {e}")

        return None