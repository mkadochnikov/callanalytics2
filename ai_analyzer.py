#!/usr/bin/env python3
"""
Класс для локального ИИ анализа текста с поддержкой анализа возражений
Интегрирован с NLU моделями (Natasha + DeepPavlov)
"""

from config import *
import pickle
from pathlib import Path
import numpy as np

# Импортируем продвинутый NLU анализатор
try:
    from advanced_nlu import AdvancedNLUAnalyzer

    NLU_AVAILABLE = True
except ImportError:
    NLU_AVAILABLE = False
    logger.warning("NLU модуль не найден. Используется базовый анализ.")


class LocalAIAnalyzer:
    """Класс для локального анализа текста с использованием трансформеров и NLU"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Используется устройство: {self.device}")

        # Инициализация NLU если доступен
        if NLU_AVAILABLE:
            try:
                self.nlu_analyzer = AdvancedNLUAnalyzer()
                self.use_nlu = True
                logger.info("✅ NLU анализатор инициализирован")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать NLU: {e}")
                self.nlu_analyzer = None
                self.use_nlu = False
        else:
            self.nlu_analyzer = None
            self.use_nlu = False

        # Инициализация базовых моделей
        self._init_models()

        # Файл для сохранения новых категорий возражений
        self.custom_objections_file = Path("bitrix_analytics/custom_objections.json")
        self.custom_objections = self._load_custom_objections()

    def _init_models(self):
        """Инициализация локальных моделей"""
        try:
            # Модель для классификации тем
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("✅ Модель классификации загружена")
        except Exception as e:
            logger.warning(f"Не удалось загрузить модель классификации: {e}")
            self.classifier = None

    def _load_custom_objections(self) -> Dict:
        """Загружает пользовательские категории возражений"""
        try:
            if self.custom_objections_file.exists():
                with open(self.custom_objections_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Ошибка загрузки пользовательских возражений: {e}")
        return {}

    def _save_custom_objections(self):
        """Сохраняет пользовательские категории возражений"""
        try:
            self.custom_objections_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.custom_objections_file, 'w', encoding='utf-8') as f:
                json.dump(self.custom_objections, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения пользовательских возражений: {e}")

    def classify_topic(self, text: str) -> str:
        """Улучшенное определение темы разговора"""
        # Если есть NLU, используем его
        if self.use_nlu and self.nlu_analyzer:
            try:
                nlu_result = self.nlu_analyzer.analyze_dialogue(text)
                return nlu_result.get("topic", "Неопределенная тема")
            except Exception as e:
                logger.warning(f"Ошибка NLU анализа темы: {e}")

        # Fallback на базовый метод
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

        return "Неопределенная тема"

    def find_objection_reason(self, text: str) -> Optional[Dict]:
        """Улучшенный поиск возражений с использованием NLU"""
        # Если есть NLU, используем его
        if self.use_nlu and self.nlu_analyzer:
            try:
                nlu_result = self.nlu_analyzer.analyze_dialogue(text)
                objections = nlu_result.get("objections", [])

                if objections:
                    # Берем самое уверенное возражение
                    main_objection = sorted(objections, key=lambda x: x.get("confidence", 0), reverse=True)[0]

                    return {
                        "objection": main_objection.get("type"),
                        "recommendation": main_objection.get("recommendation"),
                        "confidence": main_objection.get("confidence", 0),
                        "sentiment": {"sentiment": "negative", "confidence": 0.8}
                    }
            except Exception as e:
                logger.warning(f"Ошибка NLU анализа возражений: {e}")

        # Fallback на базовый метод
        if not text.strip():
            return None

        text_lower = text.lower()

        # Проверяем наличие возражения
        objection_indicators = [
            "не интересно", "не нужно", "не подходит", "не хочу", "не буду",
            "откажусь", "отказываюсь", "против", "не согласен", "не устраивает",
            "дорого", "нет времени", "подумать"
        ]

        has_objection = any(indicator in text_lower for indicator in objection_indicators)

        if not has_objection:
            return None

        # Ищем конкретную причину
        objection_scores = {}

        for objection_name, objection_data in OBJECTION_CATEGORIES.items():
            score = 0
            keywords = objection_data["keywords"]

            for keyword in keywords:
                if keyword in text_lower:
                    score += 2

            if score > 0:
                objection_scores[objection_name] = score

        if objection_scores:
            best_objection = max(objection_scores, key=objection_scores.get)
            recommendation = OBJECTION_CATEGORIES[best_objection]["recommendation"]

            return {
                "objection": best_objection,
                "recommendation": recommendation,
                "confidence": min(0.9, objection_scores[best_objection] * 0.2),
                "sentiment": {"sentiment": "negative", "confidence": 0.8}
            }

        return {
            "objection": "❓ Общее возражение",
            "recommendation": "Выяснить конкретную причину возражения",
            "confidence": 0.5,
            "sentiment": {"sentiment": "negative", "confidence": 0.8}
        }

    def extract_key_points(self, text: str) -> List[str]:
        """Извлечение ключевых моментов из текста"""
        # Если есть NLU, используем его
        if self.use_nlu and self.nlu_analyzer:
            try:
                nlu_result = self.nlu_analyzer.analyze_dialogue(text)
                key_points = nlu_result.get("key_points", [])
                if key_points:
                    return key_points
            except Exception as e:
                logger.warning(f"Ошибка NLU извлечения ключевых моментов: {e}")

        # Fallback на базовый метод
        if not text.strip():
            return []

        sentences = re.split(r'[.!?]+', text)

        key_sentences = []
        important_words = [
            'важно', 'главное', 'основное', 'нужно', 'требуется', 'проблема',
            'вопрос', 'решение', 'предложение', 'цена', 'стоимость', 'условия',
            'договор', 'заказ', 'услуга', 'продукт', 'клиент', 'компания',
            'возражение', 'не устраивает', 'не подходит', 'беспокоит'
        ]

        for sentence in sentences[:10]:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 200:
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in important_words):
                    key_sentences.append(sentence)

        return key_sentences[:3]

    def analyze_full_dialogue(self, transcript: str, call_info: Dict = None) -> Dict:
        """Полный анализ диалога с NLU"""
        if self.use_nlu and self.nlu_analyzer:
            try:
                # Полный NLU анализ
                nlu_result = self.nlu_analyzer.analyze_dialogue(transcript, call_info)

                # Преобразуем для совместимости
                return {
                    "topic": nlu_result.get("topic", "Неопределенная тема"),
                    "key_points": nlu_result.get("key_points", []),
                    "objections": nlu_result.get("objections", []),
                    "objection_reason": self._get_main_objection(nlu_result.get("objections", [])),
                    "objection_recommendation": self._get_main_recommendation(nlu_result.get("objections", [])),
                    "rejection_reason": self._get_main_objection(nlu_result.get("objections", [])),
                    "entities": nlu_result.get("entities", {}),
                    "sales_stage": nlu_result.get("sales_stage"),
                    "call_result": nlu_result.get("call_result", {}),
                    "business_data": nlu_result.get("business_data", {}),
                    "emotional_context": nlu_result.get("emotional_context", {})
                }
            except Exception as e:
                logger.error(f"Ошибка полного NLU анализа: {e}")

        # Fallback на базовый анализ
        topic = self.classify_topic(transcript)
        objection_result = self.find_objection_reason(transcript)
        key_points = self.extract_key_points(transcript)

        return {
            "topic": topic,
            "key_points": key_points,
            "objection_reason": objection_result["objection"] if objection_result else None,
            "objection_recommendation": objection_result["recommendation"] if objection_result else None,
            "rejection_reason": objection_result["objection"] if objection_result else None,
            "sentiment": objection_result["sentiment"] if objection_result else {"sentiment": "neutral",
                                                                                 "confidence": 0.5}
        }

    def _get_main_objection(self, objections: List[Dict]) -> Optional[str]:
        """Получает главное возражение из списка"""
        if not objections:
            return None
        sorted_objections = sorted(objections, key=lambda x: x.get("confidence", 0), reverse=True)
        return sorted_objections[0].get("type") if sorted_objections else None

    def _get_main_recommendation(self, objections: List[Dict]) -> Optional[str]:
        """Получает главную рекомендацию"""
        if not objections:
            return None
        sorted_objections = sorted(objections, key=lambda x: x.get("confidence", 0), reverse=True)
        return sorted_objections[0].get("recommendation") if sorted_objections else None

    def get_all_objection_categories(self) -> Dict:
        """Возвращает все категории возражений"""
        all_categories = OBJECTION_CATEGORIES.copy()
        all_categories.update(self.custom_objections)
        return all_categories

    def analyze_sentiment(self, text: str) -> Dict:
        """Анализ тональности (заглушка для совместимости)"""
        return {"sentiment": "neutral", "confidence": 0.5}

    # Метод для обратной совместимости
    def find_rejection_reason(self, text: str) -> Optional[str]:
        """Устаревший метод для обратной совместимости"""
        objection_result = self.find_objection_reason(text)
        if objection_result:
            return objection_result["objection"]
        return None