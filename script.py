#!/usr/bin/env python3
"""
Система анализа звонков Bitrix24 с ЛОКАЛЬНЫМИ моделями ИИ
Полный цикл: загрузка -> транскрибация (Whisper) -> анализ (локальная модель) -> отчет
"""

# Конфигурация и импорты для системы анализа звонков Bitrix24

import os
import json
import hashlib
import requests
import datetime
import pytz
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd
import plotly.express as px
import whisper
import torch
from transformers import pipeline
import re

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Конфигурация
PORTAL_TIMEZONE = 'Europe/Moscow'
WHISPER_MODEL_SIZE = "medium"

# Расширенные темы звонков с ключевыми словами
CALL_TOPICS = {
    "Продажи": ["продажи", "предложение", "коммерческий", "покупка", "заказ", "прайс", "цена", "стоимость", "купить", "приобрести", "товар", "услуга", "скидка", "акция"],
    "Техподдержка": ["проблема", "ошибка", "не работает", "сломался", "поломка", "техническая", "поддержка", "помощь", "настройка", "установка", "баг", "глюк"],
    "Консультация": ["вопрос", "как", "можно ли", "расскажите", "объясните", "консультация", "информация", "уточнить", "узнать", "подскажите"],
    "Жалоба": ["жалоба", "недоволен", "плохо", "ужасно", "возврат", "претензия", "некачественно", "обман", "мошенники", "верните деньги"],
    "Партнерство": ["партнерство", "сотрудничество", "совместный", "партнер", "коллаборация", "взаимовыгодный", "предложение о сотрудничестве"],
    "Холодные продажи": ["холодный", "знакомство", "представляю компанию", "слышали о нас", "предлагаем", "заинтересованы ли"],
    "Повторное обращение": ["звонил ранее", "обещали", "договорились", "следующий этап", "продолжение", "как дела с"]
}

# Расширенные причины отказов с ключевыми фразами
REJECTION_PATTERNS = {
    "Высокая цена": ["дорого", "цена высокая", "много денег", "не по карману", "дешевле", "слишком дорого", "цена не подходит", "бюджет меньше"],
    "Нет времени": ["времени нет", "некогда", "занят", "позже позвоню", "не сейчас", "в другой раз", "нет времени разговаривать"],
    "Есть поставщик": ["уже работаем с", "есть поставщик", "другая компания", "уже заключен договор", "постоянный поставщик"],
    "Не нужна услуга": ["не нужно", "не интересует", "не актуально", "не требуется", "нет необходимости", "не подходит"],
    "Нужно подумать": ["подумать", "посоветоваться", "обсудить", "принять решение", "рассмотрим", "изучим предложение"],
    "Нет бюджета": ["нет денег", "нет бюджета", "нет средств", "финансовые трудности", "кризис", "экономим"],
    "Плохие условия": ["условия не подходят", "не устраивают условия", "другие требования", "не согласны с условиями"],
    "Недоверие": ["не доверяю", "мошенники", "обман", "развод на деньги", "сомневаюсь", "не верю"],
    "Конкуренты лучше": ["у конкурентов", "другие предлагают", "лучше условия", "дешевле у других"]
}

# Класс для локального ИИ анализа текста

from config import *


class LocalAIAnalyzer:
    """Класс для локального анализа текста с использованием трансформеров"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Используется устройство: {self.device}")
        self._init_models()

    def _init_models(self):
        """Инициализация локальных моделей"""
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("✅ Модель классификации загружена")
        except Exception as e:
            logger.warning(f"Не удалось загрузить модель классификации: {e}")
            self.classifier = None

    def classify_topic(self, text: str) -> str:
        """Улучшенное определение темы разговора"""
        if not text.strip():
            return "Общий разговор"

        text_lower = text.lower()

        # Проверяем по ключевым словам
        topic_scores = {}
        for topic, keywords in CALL_TOPICS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                topic_scores[topic] = score

        # Если нашли совпадения по ключевым словам
        if topic_scores:
            best_topic = max(topic_scores, key=topic_scores.get)
            return best_topic

        # Используем ИИ классификацию
        if self.classifier:
            try:
                text_sample = text[:1000]
                result = self.classifier(text_sample, list(CALL_TOPICS.keys()))

                if result and 'labels' in result and len(result['labels']) > 0:
                    confidence = result['scores'][0]
                    if confidence > 0.2:
                        return result['labels'][0]
            except Exception as e:
                logger.error(f"Ошибка ИИ классификации: {e}")

        # Дополнительная эвристика
        if len(text) < 50:
            return "Короткий разговор"
        elif any(word in text_lower for word in ["здравствуйте", "добрый день", "привет"]):
            return "Общий разговор"
        else:
            return "Неопределенная тема"

    def find_rejection_reason(self, text: str) -> Optional[str]:
        """Улучшенный поиск причины отказа"""
        if not text.strip():
            return None

        text_lower = text.lower()

        # Проверяем наличие отказных фраз
        rejection_indicators = [
            "нет", "не", "отказ", "не интересно", "не нужно", "не подходит",
            "не хочу", "не буду", "откажусь", "спасибо, не надо"
        ]

        has_rejection = any(indicator in text_lower for indicator in rejection_indicators)

        if not has_rejection:
            return None

        # Ищем конкретную причину отказа
        reason_scores = {}
        for reason, patterns in REJECTION_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if pattern in text_lower:
                    # Учитываем контекст
                    pattern_pos = text_lower.find(pattern)
                    context = text_lower[max(0, pattern_pos - 100):pattern_pos + 100]
                    if any(reject_word in context for reject_word in rejection_indicators):
                        score += 2  # Высокий вес если рядом с отказом
                    else:
                        score += 1  # Низкий вес если просто упоминание

            if score > 0:
                reason_scores[reason] = score

        if reason_scores:
            best_reason = max(reason_scores, key=reason_scores.get)
            return best_reason

        return "Общий отказ"

    def extract_key_points(self, text: str) -> List[str]:
        """Извлечение ключевых моментов из текста"""
        if not text.strip():
            return []

        sentences = re.split(r'[.!?]+', text)

        key_sentences = []
        important_words = [
            'важно', 'главное', 'основное', 'нужно', 'требуется', 'проблема',
            'вопрос', 'решение', 'предложение', 'цена', 'стоимость', 'условия',
            'договор', 'заказ', 'услуга', 'продукт', 'клиент', 'компания'
        ]

        for sentence in sentences[:10]:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 200:
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in important_words):
                    key_sentences.append(sentence)

        return key_sentences[:3]

# API методы для работы с Bitrix24

from config import *


class BitrixAPI:
    """Класс для работы с API Bitrix24"""

    def __init__(self):
        self.webhook_url = os.getenv('BITRIX_WEBHOOK_URL', '').rstrip('/')
        self.username = os.getenv('BITRIX_USERNAME', '')
        self.password = os.getenv('BITRIX_PASSWORD', '')

        self.session = requests.Session()
        self.authenticated = False

    def make_api_call(self, method: str, params: Dict = None) -> Optional[Dict]:
        """Выполняет API вызов к Bitrix24"""
        if not self.webhook_url:
            return None

        url = f"{self.webhook_url}/{method}"

        try:
            if params is None:
                params = {}

            response = requests.post(url, json=params, timeout=30)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Ошибка API {method}: {e}")
            return None

    def authenticate_bitrix(self) -> bool:
        """Авторизуется в Bitrix24"""
        if not self.username or not self.password:
            return False

        logger.info("Авторизация в Bitrix24...")

        base_url = self.webhook_url.split('/rest/')[0]
        auth_url = f"{base_url}/auth/"

        try:
            auth_page = self.session.get(auth_url, timeout=30)

            if auth_page.status_code == 200:
                auth_data = {
                    'USER_LOGIN': self.username,
                    'USER_PASSWORD': self.password
                }

                # Парсим скрытые поля
                soup = BeautifulSoup(auth_page.content, 'html.parser')
                for hidden_input in soup.find_all('input', type='hidden'):
                    name = hidden_input.get('name')
                    value = hidden_input.get('value')
                    if name and value:
                        auth_data[name] = value

                login_response = self.session.post(auth_url, data=auth_data, timeout=30)

                if login_response.status_code == 200:
                    if 'logout' in login_response.text.lower() or 'выйти' in login_response.text.lower():
                        logger.info("Авторизация успешна!")
                        self.authenticated = True
                        return True

            return False

        except Exception as e:
            logger.error(f"Ошибка авторизации: {e}")
            return False

    def get_all_calls_for_day(self, target_day: datetime.datetime) -> List[Dict]:
        """Получает все звонки за день"""
        all_calls = []
        start = 0

        tz = pytz.timezone(PORTAL_TIMEZONE)
        start_date_utc = tz.localize(target_day).astimezone(pytz.utc)
        end_date_utc = tz.localize(target_day.replace(hour=23, minute=59, second=59)).astimezone(pytz.utc)

        logger.info(f"Загрузка звонков за {target_day.strftime('%d.%m.%Y')}...")

        while True:
            params = {
                'filter': {
                    '>=CALL_START_DATE': start_date_utc.isoformat(),
                    '<=CALL_START_DATE': end_date_utc.isoformat()
                },
                'start': start
            }

            data = self.make_api_call("voximplant.statistic.get", params)

            if data and 'result' in data and data['result']:
                batch_calls = data['result']
                all_calls.extend(batch_calls)

                if len(batch_calls) < 50:
                    break
                start += 50
            else:
                break

        logger.info(f"Найдено {len(all_calls)} звонков")
        return all_calls

    def get_user_names(self, user_ids: set) -> Dict[str, str]:
        """Получает имена пользователей"""
        if not user_ids:
            return {}

        data = self.make_api_call("user.get", {'ID': list(user_ids)})

        if data and 'result' in data:
            user_names = {}
            for user in data['result']:
                name = f"{user.get('NAME', '')} {user.get('LAST_NAME', '')}".strip()
                user_names[user['ID']] = name or f"User_{user['ID']}"
            return user_names

        return {}

    def determine_call_direction(self, call: Dict) -> str:
        """Улучшенное определение направления звонка"""
        call_type = call.get('CALL_TYPE', '')

        # Bitrix24 коды направлений
        if call_type == 1 or call_type == '1':
            return 'incoming'
        elif call_type == 2 or call_type == '2':
            return 'outgoing'

        # Дополнительная логика по полям
        call_direction = call.get('CALL_DIRECTION', '').lower()
        if 'in' in call_direction:
            return 'incoming'
        elif 'out' in call_direction:
            return 'outgoing'

        # Анализ по времени звонка
        phone_number = call.get('PHONE_NUMBER', '')
        portal_user_id = call.get('PORTAL_USER_ID', '')

        if phone_number and portal_user_id:
            call_start_date = call.get('CALL_START_DATE', '')
            if call_start_date:
                try:
                    dt = datetime.datetime.fromisoformat(call_start_date.replace('Z', '+00:00'))
                    hour = dt.hour
                    if 9 <= hour <= 18:  # Рабочие часы
                        return 'incoming'
                    else:
                        return 'outgoing'
                except:
                    pass

        return 'unknown'

# Класс для обработки аудио и транскрибации

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

# Класс для управления данными и кешированием

from config import *


class DataManager:
    """Класс для управления данными и кешированием"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.cache_dir = base_dir / "cache"
        self.analysis_dir = base_dir / "analysis"

        # Создаем папки
        for dir_path in [self.cache_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_date_hash(self, target_date: datetime.datetime) -> str:
        """Создает хеш для даты для кеширования"""
        date_str = target_date.strftime("%Y-%m-%d")
        return hashlib.md5(date_str.encode()).hexdigest()[:8]

    def is_data_cached(self, target_date: datetime.datetime) -> bool:
        """Проверяет, есть ли кешированные данные за дату"""
        date_str = target_date.strftime("%Y-%m-%d")
        cache_file = self.cache_dir / f"calls_{date_str}.json"
        return cache_file.exists()

    def save_calls_to_cache(self, target_date: datetime.datetime, calls: List[Dict]):
        """Сохраняет звонки в кеш"""
        date_str = target_date.strftime("%Y-%m-%d")
        cache_file = self.cache_dir / f"calls_{date_str}.json"

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(calls, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Данные сохранены в кеш: {cache_file}")

    def load_calls_from_cache(self, target_date: datetime.datetime) -> List[Dict]:
        """Загружает звонки из кеша"""
        date_str = target_date.strftime("%Y-%m-%d")
        cache_file = self.cache_dir / f"calls_{date_str}.json"

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                calls = json.load(f)
            logger.info(f"Данные загружены из кеша: {cache_file}")
            return calls
        except Exception as e:
            logger.error(f"Ошибка загрузки из кеша: {e}")
            return []

    def save_analysis(self, analysis: Dict, filename: str, date_str: str):
        """Сохраняет результат анализа"""
        analysis_date_dir = self.analysis_dir / date_str
        analysis_date_dir.mkdir(exist_ok=True)

        analysis_file = analysis_date_dir / f"{filename}_analysis.json"

        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

    def load_analysis(self, filename: str, date_str: str) -> Optional[Dict]:
        """Загружает результат анализа"""
        analysis_file = self.analysis_dir / date_str / f"{filename}_analysis.json"

        if analysis_file.exists():
            try:
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Ошибка загрузки анализа: {e}")

        return None

    def get_test_data(self, target_date: datetime.datetime) -> List[Dict]:
        """Генерирует тестовые данные для демонстрации"""
        test_transcripts = [
            "Здравствуйте! Меня интересует ваш продукт. Какая цена? Понятно, но это слишком дорого для нас. Спасибо, пока.",
            "Добрый день! У нас проблема с системой. Не работает модуль отчетов. Нужна срочная помощь технической поддержки.",
            "Привет! Хочу заказать вашу услугу. Когда можете начать? Отлично, договорились!",
            "Звоню по поводу партнерства. Наша компания готова к сотрудничеству. Давайте встретимся.",
            "У нас жалоба на качество обслуживания. Очень недовольны. Требуем возврат денег!"
        ]

        test_calls = []
        for i in range(5):
            call_time = target_date + datetime.timedelta(hours=9 + i, minutes=i * 10)

            test_call = {
                'ID': f'test_{i + 1}',
                'CRM_ACTIVITY_ID': f'activity_{i + 1}',
                'CALL_START_DATE': call_time.isoformat(),
                'CALL_TYPE': 1 if i % 2 == 0 else 2,
                'PHONE_NUMBER': f'+7900123456{i}',
                'PORTAL_USER_ID': '1',
                'user_name': 'Тестовый Менеджер',
                'CALL_DURATION': 120 + i * 30,
                'test_data': True,
                'transcript': test_transcripts[i],
                'call_direction': 'incoming' if i % 2 == 0 else 'outgoing'
            }

            test_calls.append(test_call)

        logger.info(f"Сгенерировано {len(test_calls)} тестовых звонков")
        return test_calls

# Основной класс для анализа звонков Bitrix24

from config import *
from ai_analyzer import LocalAIAnalyzer
from bitrix_api import BitrixAPI
from audio_processor import AudioProcessor
from data_manager import DataManager


class BitrixCallAnalyzer:
    """Основной класс для анализа звонков Bitrix24 с локальными моделями"""

    def __init__(self):
        # Структура папок
        self.base_dir = Path("bitrix_analytics")
        self.base_dir.mkdir(exist_ok=True)
        self.reports_dir = self.base_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)

        # Инициализация компонентов
        self.ai_analyzer = LocalAIAnalyzer()
        self.bitrix_api = BitrixAPI()
        self.audio_processor = AudioProcessor(self.base_dir)
        self.data_manager = DataManager(self.base_dir)

        logger.info("Инициализация BitrixCallAnalyzer завершена")

    def analyze_transcript(self, transcript: str, call_info: Dict) -> Dict:
        """Анализирует транскрипцию звонка с помощью локальных моделей"""
        if not transcript or transcript in ["Ошибка транскрибации", "Не удалось распознать речь"]:
            return {
                "topic": "Неопределенная тема",
                "key_points": [],
                "rejection_reason": None
            }

        try:
            logger.info("Анализ транскрипции с помощью локальных моделей...")

            # Определяем тему (улучшенный алгоритм)
            topic = self.ai_analyzer.classify_topic(transcript)

            # Ищем причину отказа (улучшенный алгоритм)
            rejection_reason = self.ai_analyzer.find_rejection_reason(transcript)

            # Извлекаем ключевые моменты
            key_points = self.ai_analyzer.extract_key_points(transcript)

            result = {
                "topic": topic,
                "key_points": key_points,
                "rejection_reason": rejection_reason
            }

            logger.info(f"Анализ завершен: тема={topic}, отказ={rejection_reason}")
            return result

        except Exception as e:
            logger.error(f"Ошибка анализа транскрипции: {e}")
            return {
                "topic": "Ошибка анализа",
                "key_points": [],
                "rejection_reason": None
            }

    def process_calls_for_date(self, target_date: datetime.datetime) -> List[Dict]:
        """Обрабатывает звонки за дату - полный цикл"""
        date_str = target_date.strftime("%Y-%m-%d")

        # Проверяем кеш
        if self.data_manager.is_data_cached(target_date):
            logger.info(f"Используем кешированные данные за {date_str}")
            return self.data_manager.load_calls_from_cache(target_date)

        # Загружаем новые данные
        if not self.bitrix_api.webhook_url:
            logger.warning("Bitrix24 не настроен, используем тестовые данные")
            calls = self.data_manager.get_test_data(target_date)
        else:
            # Авторизуемся
            if self.bitrix_api.username and self.bitrix_api.password:
                self.bitrix_api.authenticate_bitrix()

            # Получаем звонки
            calls = self.bitrix_api.get_all_calls_for_day(target_date)

            if not calls:
                return []

            # Получаем имена пользователей
            user_ids = {call['PORTAL_USER_ID'] for call in calls if call.get('PORTAL_USER_ID')}
            user_names = self.bitrix_api.get_user_names(user_ids)

            # Добавляем имена к звонкам и определяем направление
            for call in calls:
                user_id = call.get('PORTAL_USER_ID', '')
                call['user_name'] = user_names.get(user_id, 'Unknown')
                call['call_direction'] = self.bitrix_api.determine_call_direction(call)

            # Извлекаем аудиозаписи если авторизованы
            if self.bitrix_api.authenticated:
                calls = self._extract_audio_recordings(calls, user_names, date_str)

        # Транскрибация и анализ
        for call in calls:
            if 'audio_file' in call:
                # Транскрибация
                filename = Path(call['audio_file']).stem
                existing_transcript = self.audio_processor.load_transcript(filename, date_str)

                if existing_transcript:
                    call['transcript'] = existing_transcript
                else:
                    logger.info(f"Транскрибируем: {call.get('audio_filename', 'аудио')}")
                    transcript = self.audio_processor.transcribe_audio(call['audio_file'])
                    call['transcript'] = transcript
                    self.audio_processor.save_transcript(transcript, filename, date_str)

                # Анализ
                existing_analysis = self.data_manager.load_analysis(filename, date_str)

                if existing_analysis:
                    call['analysis'] = existing_analysis
                else:
                    logger.info(f"Анализируем: {call.get('audio_filename', 'транскрипцию')}")
                    analysis = self.analyze_transcript(call['transcript'], call)
                    call['analysis'] = analysis
                    self.data_manager.save_analysis(analysis, filename, date_str)

            # Для тестовых данных добавляем анализ
            elif call.get('transcript') and not call.get('analysis'):
                call['analysis'] = self.analyze_transcript(call['transcript'], call)

        # Сохраняем в кеш
        self.data_manager.save_calls_to_cache(target_date, calls)

        return calls

    def _extract_audio_recordings(self, calls: List[Dict], user_names: Dict[str, str], date_str: str) -> List[Dict]:
        """Извлекает аудиозаписи звонков с улучшенным определением направления"""
        logger.info("Извлечение аудиозаписей...")

        activity_ids = [call.get('CRM_ACTIVITY_ID') for call in calls if call.get('CRM_ACTIVITY_ID')]

        if not activity_ids:
            return calls

        calls_map = {call.get('CRM_ACTIVITY_ID'): call for call in calls if call.get('CRM_ACTIVITY_ID')}

        # Поиск активностей с файлами
        for aid in activity_ids:
            activity_data = self.bitrix_api.make_api_call("crm.activity.get", {"id": aid})
            if activity_data and 'result' in activity_data:
                activity = activity_data['result']
                call = calls_map.get(aid)
                if call and 'FILES' in activity and activity['FILES']:
                    # Формируем имя файла
                    call_time = call.get('CALL_START_DATE', '')
                    if call_time:
                        try:
                            utc_time = datetime.datetime.fromisoformat(call_time.replace('Z', '+00:00'))
                            local_time = utc_time.astimezone(pytz.timezone(PORTAL_TIMEZONE))
                            time_str = local_time.strftime('%H-%M-%S')
                        except:
                            time_str = "unknown-time"
                    else:
                        time_str = "unknown-time"

                    user_id = call.get('PORTAL_USER_ID', '')
                    user_name = user_names.get(user_id, f'user_{user_id}').replace(' ', '_')

                    phone_number = call.get('PHONE_NUMBER', 'unknown_number')
                    clean_phone = ''.join(filter(str.isdigit, phone_number))

                    direction = call.get('call_direction', 'unknown')
                    filename = f"{date_str}_{time_str}_{direction}_{user_name}_{clean_phone}_id{aid}"

                    for file_info in activity['FILES']:
                        file_path = self.audio_processor.download_audio_file(
                            file_info, filename, date_str, self.bitrix_api.session
                        )
                        if file_path:
                            call['audio_file'] = file_path
                            call['audio_filename'] = f"{filename}.mp3"
                            break

        return calls

    def generate_pdf_report(self, all_calls: List[Dict], date_range: str) -> Optional[bytes]:
        """Генерирует PDF отчет"""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            import io

            # Создаем PDF в памяти
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)

            # Стили
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=30,
                alignment=1
            )

            story = []

            # Заголовок
            story.append(Paragraph("Отчет по звонкам Bitrix24", title_style))
            story.append(Paragraph(f"Период: {date_range}", styles['Normal']))
            story.append(Paragraph(f"Создан: {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')}", styles['Normal']))
            story.append(Spacer(1, 20))

            # Общая статистика
            total_calls = len(all_calls)
            incoming_calls = sum(1 for call in all_calls if call.get('call_direction') == 'incoming')
            outgoing_calls = sum(1 for call in all_calls if call.get('call_direction') == 'outgoing')

            stats_data = [
                ['Показатель', 'Значение'],
                ['Всего звонков', str(total_calls)],
                ['Входящие звонки', str(incoming_calls)],
                ['Исходящие звонки', str(outgoing_calls)],
                ['С аудиозаписью', str(sum(1 for call in all_calls if 'audio_file' in call))]
            ]

            stats_table = Table(stats_data)
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(Paragraph("Общая статистика", styles['Heading2']))
            story.append(stats_table)
            story.append(Spacer(1, 20))

            # Темы звонков
            topics = {}
            for call in all_calls:
                analysis = call.get('analysis', {})
                topic = analysis.get('topic', 'Неопределенная тема')
                topics[topic] = topics.get(topic, 0) + 1

            if topics:
                story.append(Paragraph("Темы звонков", styles['Heading2']))
                topic_data = [['Тема', 'Количество']]
                for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True):
                    topic_data.append([topic, str(count)])

                topic_table = Table(topic_data)
                topic_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))

                story.append(topic_table)

            # Создаем PDF
            doc.build(story)

            # Возвращаем байты PDF
            pdf_bytes = buffer.getvalue()
            buffer.close()

            return pdf_bytes

        except ImportError:
            logger.error("Для экспорта PDF установите: pip install reportlab")
            return None
        except Exception as e:
            logger.error(f"Ошибка создания PDF: {e}")
            return None

# Streamlit веб-интерфейс для системы анализа звонков

from config import *
from main_analyzer import BitrixCallAnalyzer


def main():
    """Основная функция со Streamlit интерфейсом"""
    st.set_page_config(
        page_title="Bitrix24 Local AI Analytics",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🤖 Bitrix24 Local AI Analytics")
    st.markdown("**Полностью локальная система анализа звонков с ИИ** • Whisper + Transformers")

    # Проверяем доступность CUDA
    device_info = "🔥 CUDA GPU" if torch.cuda.is_available() else "💻 CPU"
    st.sidebar.info(f"Устройство обработки: {device_info}")

    # Инициализация analyzer
    if 'analyzer' not in st.session_state:
        with st.spinner("Загрузка моделей ИИ..."):
            st.session_state.analyzer = BitrixCallAnalyzer()

    analyzer = st.session_state.analyzer

    # Sidebar с настройками
    with st.sidebar:
        st.header("⚙️ Настройки подключения")

        # Настройки Bitrix24
        st.subheader("🏢 Bitrix24")
        webhook_url = st.text_input("Webhook URL", value=analyzer.bitrix_api.webhook_url, type="password")
        username = st.text_input("Username", value=analyzer.bitrix_api.username)
        password = st.text_input("Password", type="password", value=analyzer.bitrix_api.password)

        if st.button("💾 Сохранить настройки"):
            # Обновляем настройки
            analyzer.bitrix_api.webhook_url = webhook_url
            analyzer.bitrix_api.username = username
            analyzer.bitrix_api.password = password
            st.success("✅ Настройки сохранены!")

        st.markdown("---")

        # Информация о моделях
        st.subheader("🤖 Статус локальных моделей")

        whisper_status = "✅ Загружена" if analyzer.audio_processor.whisper_model else "❌ Ошибка"
        st.write(f"**Whisper:** {whisper_status}")

        classifier_status = "✅ Загружена" if analyzer.ai_analyzer.classifier else "❌ Ошибка"
        st.write(f"**Классификация тем:** {classifier_status}")

    # Основной интерфейс
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📅 Выбор периода анализа")

        # Выбор даты
        date_option = st.radio(
            "Выберите период:",
            ["Один день", "Диапазон дат"],
            horizontal=True
        )

        if date_option == "Один день":
            selected_date = st.date_input(
                "Выберите дату:",
                value=datetime.date.today() - datetime.timedelta(days=1)
            )
            start_date = end_date = selected_date
        else:
            col_start, col_end = st.columns(2)
            with col_start:
                start_date = st.date_input(
                    "Начальная дата:",
                    value=datetime.date.today() - datetime.timedelta(days=7)
                )
            with col_end:
                end_date = st.date_input(
                    "Конечная дата:",
                    value=datetime.date.today() - datetime.timedelta(days=1)
                )

    with col2:
        st.header("🚀 Действия")

        if st.button("🤖 Загрузить и проанализировать", type="primary", use_container_width=True):
            if start_date <= end_date:
                with st.spinner("Обработка с локальными ИИ моделями..."):
                    all_calls = []

                    # Прогресс бар
                    date_range = pd.date_range(start_date, end_date)
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, date in enumerate(date_range):
                        status_text.info(
                            f"🔄 Обработка: {date.strftime('%d.%m.%Y')} (день {i + 1} из {len(date_range)})")

                        target_datetime = datetime.datetime.combine(date.date(), datetime.time())
                        calls = analyzer.process_calls_for_date(target_datetime)
                        all_calls.extend(calls)

                        progress_bar.progress((i + 1) / len(date_range))

                    st.session_state.all_calls = all_calls
                    status_text.success(f"✅ Обработано {len(all_calls)} звонков с помощью локальных ИИ моделей")
            else:
                st.error("❌ Начальная дата должна быть меньше конечной")

    # Показываем результаты анализа
    show_analysis_results()


def show_analysis_results():
    """Показывает результаты анализа"""
    if 'all_calls' not in st.session_state or not st.session_state.all_calls:
        return

    st.header("📊 Статистика по обработанным данным")

    calls_data = st.session_state.all_calls

    # Метрики
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Всего звонков", len(calls_data))

    with col2:
        incoming = sum(1 for call in calls_data if call.get('call_direction') == 'incoming')
        st.metric("Входящие", incoming)

    with col3:
        outgoing = sum(1 for call in calls_data if call.get('call_direction') == 'outgoing')
        st.metric("Исходящие", outgoing)

    with col4:
        unknown = sum(1 for call in calls_data if call.get('call_direction') == 'unknown')
        st.metric("Неопределенные", unknown)

    with col5:
        with_audio = sum(1 for call in calls_data if 'audio_file' in call)
        st.metric("С аудиозаписью", with_audio)

    with col6:
        analyzed = sum(1 for call in calls_data if 'analysis' in call)
        st.metric("Проанализировано ИИ", analyzed)

    # Сводные таблицы
    show_summary_tables(calls_data)

    # Детализация
    show_call_details(calls_data)


def show_summary_tables(calls_data):
    """Показывает сводные таблицы"""
    st.header("📊 Сводные таблицы ИИ анализа")

    col1, col2 = st.columns(2)

    with col1:
        # Таблица тем
        topic_data = {}
        for call in calls_data:
            analysis = call.get('analysis', {})
            topic = analysis.get('topic', 'Неопределенная тема')
            topic_data[topic] = topic_data.get(topic, 0) + 1

        if topic_data:
            st.subheader("🎯 Темы звонков")
            topic_df = pd.DataFrame([
                {'Тема': topic, 'Количество': count, 'Процент': f"{count / len(calls_data) * 100:.1f}%"}
                for topic, count in sorted(topic_data.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(topic_df, use_container_width=True, hide_index=True)

    with col2:
        # Таблица отказов
        rejection_data = {}
        for call in calls_data:
            analysis = call.get('analysis', {})
            rejection = analysis.get('rejection_reason')
            if rejection:
                rejection_data[rejection] = rejection_data.get(rejection, 0) + 1

        if rejection_data:
            st.subheader("❌ Причины отказов")
            rejection_df = pd.DataFrame([
                {'Причина отказа': reason, 'Количество': count,
                 'Процент': f"{count / len([c for c in calls_data if c.get('analysis', {}).get('rejection_reason')]) * 100:.1f}%"}
                for reason, count in sorted(rejection_data.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(rejection_df, use_container_width=True, hide_index=True)
        else:
            st.subheader("❌ Причины отказов")
            st.info("Отказы не обнаружены в анализируемых звонках")


def show_call_details(calls_data):
    """Показывает детализацию звонков"""
    st.header("🔍 Детализация звонков с ИИ анализом")

    # Подготавливаем данные для таблицы
    table_data = []
    for call in calls_data:
        analysis = call.get('analysis', {})

        # Форматируем время
        call_time = call.get('CALL_START_DATE', '')
        if call_time and 'T' in call_time:
            try:
                dt = datetime.datetime.fromisoformat(call_time.replace('Z', '+00:00'))
                formatted_time = dt.strftime('%d.%m.%Y %H:%M')
            except:
                formatted_time = call_time
        else:
            formatted_time = call_time

        table_data.append({
            'Дата/время': formatted_time,
            'Менеджер': call.get('user_name', ''),
            'Телефон': call.get('PHONE_NUMBER', ''),
            'Тип': call.get('call_direction', 'unknown').replace('incoming', '📞 Входящий').replace('outgoing',
                                                                                                   '📱 Исходящий').replace(
                'unknown', '❓ Неопределенный'),
            'Длительность': f"{call.get('CALL_DURATION', 0)} сек",
            'Тема (ИИ)': analysis.get('topic', 'Неопределенная тема'),
            'Причина отказа (ИИ)': analysis.get('rejection_reason', ''),
            'Транскрипция': '✅ Есть' if call.get('transcript') else '❌ Нет'
        })

    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)

        # Показываем примеры транскрипций
        show_transcript_examples(calls_data)


def show_transcript_examples(calls_data):
    """Показывает примеры транскрипций"""
    st.subheader("📝 Примеры транскрипций и анализа")

    calls_with_transcripts = [call for call in calls_data if call.get('transcript') and call.get('analysis')]

    if calls_with_transcripts:
        selected_call = st.selectbox(
            "Выберите звонок для просмотра:",
            range(min(10, len(calls_with_transcripts))),
            format_func=lambda
                x: f"Звонок {x + 1}: {calls_with_transcripts[x].get('user_name', 'Unknown')} - {calls_with_transcripts[x]['analysis'].get('topic', 'Неопределенная тема')}"
        )

        if selected_call is not None:
            call = calls_with_transcripts[selected_call]
            analysis = call.get('analysis', {})

            col1, col2 = st.columns([2, 1])

            with col1:
                st.write("**Транскрипция:**")
                st.text_area("Текст разговора", call.get('transcript', ''), height=200, disabled=True,
                             label_visibility="collapsed")

            with col2:
                st.write("**ИИ Анализ:**")
                st.write(f"**Тема:** {analysis.get('topic', 'Неопределенная тема')}")

                if analysis.get('rejection_reason'):
                    st.write(f"**Причина отказа:** {analysis['rejection_reason']}")
                else:
                    st.write("**Причина отказа:** Не выявлена")

                key_points = analysis.get('key_points', [])
                if key_points:
                    st.write("**Ключевые моменты:**")
                    for i, point in enumerate(key_points, 1):
                        st.write(f"{i}. {point}")
                else:
                    st.write("**Ключевые моменты:** Не выявлены")


if __name__ == "__main__":
    main()
