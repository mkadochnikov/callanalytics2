#!/usr/bin/env python3
"""
Класс для локального ИИ анализа текста с поддержкой анализа возражений
"""

from config import *
import pickle
from pathlib import Path
import numpy as np


class LocalAIAnalyzer:
    """Класс для локального анализа текста с использованием трансформеров"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Используется устройство: {self.device}")
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

        try:
            # Модель для анализа тональности и возражений
            self.sentiment_model_name = "sismetanin/rubert-ru-sentiment-rureviews"
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_name)

            # Переносим модель на нужное устройство
            self.sentiment_model.to(self.device)
            self.sentiment_model.eval()

            logger.info("✅ Модель анализа тональности sismetanin/rubert-ru-sentiment-rureviews загружена")
        except Exception as e:
            logger.warning(f"Не удалось загрузить модель sentiment: {e}")
            self.sentiment_tokenizer = None
            self.sentiment_model = None

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

    def analyze_sentiment(self, text: str) -> Dict:
        """Анализирует тональность текста с помощью RuBERT модели"""
        if not text.strip() or not self.sentiment_model or not self.sentiment_tokenizer:
            return {"sentiment": "neutral", "confidence": 0.0}

        try:
            # Подготовка текста
            inputs = self.sentiment_tokenizer(
                text[:512],  # Ограничиваем длину
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )

            # Перенос на устройство
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Предсказание
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Получение результата
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = torch.max(predictions).item()

            # Маппинг классов (зависит от конкретной модели)
            sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
            sentiment = sentiment_map.get(predicted_class, "neutral")

            return {
                "sentiment": sentiment,
                "confidence": float(confidence)
            }

        except Exception as e:
            logger.error(f"Ошибка анализа тональности: {e}")
            return {"sentiment": "neutral", "confidence": 0.0}

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

    def find_objection_reason(self, text: str) -> Optional[Dict]:
        """Улучшенный поиск возражений с использованием sentiment анализа"""
        if not text.strip():
            return None

        text_lower = text.lower()

        # Анализируем тональность для определения негативного контекста
        sentiment_result = self.analyze_sentiment(text)

        # Проверяем наличие отрицательных маркеров
        objection_indicators = [
            "нет", "не", "отказ", "не интересно", "не нужно", "не подходит",
            "не хочу", "не буду", "откажусь", "спасибо, не надо", "отказываюсь",
            "не согласен", "не устраивает", "против"
        ]

        has_objection = any(indicator in text_lower for indicator in objection_indicators)

        # Учитываем тональность - негативная тональность может указывать на возражение
        if sentiment_result["sentiment"] == "negative" and sentiment_result["confidence"] > 0.6:
            has_objection = True

        if not has_objection:
            return None

        # Ищем конкретную категорию возражения
        objection_scores = {}

        # Проверяем основные категории
        for objection_name, objection_data in OBJECTION_CATEGORIES.items():
            score = 0
            keywords = objection_data["keywords"]

            for keyword in keywords:
                if keyword in text_lower:
                    # Учитываем контекст
                    keyword_pos = text_lower.find(keyword)
                    context = text_lower[max(0, keyword_pos - 100):keyword_pos + 100]

                    if any(reject_word in context for reject_word in objection_indicators):
                        score += 3  # Высокий вес если рядом с отказом
                    elif sentiment_result["sentiment"] == "negative":
                        score += 2  # Средний вес при негативной тональности
                    else:
                        score += 1  # Низкий вес если просто упоминание

            if score > 0:
                objection_scores[objection_name] = score

        # Проверяем пользовательские категории
        for custom_name, custom_data in self.custom_objections.items():
            score = 0
            keywords = custom_data.get("keywords", [])

            for keyword in keywords:
                if keyword in text_lower:
                    keyword_pos = text_lower.find(keyword)
                    context = text_lower[max(0, keyword_pos - 100):keyword_pos + 100]

                    if any(reject_word in context for reject_word in objection_indicators):
                        score += 3
                    elif sentiment_result["sentiment"] == "negative":
                        score += 2
                    else:
                        score += 1

            if score > 0:
                objection_scores[custom_name] = score

        if objection_scores:
            best_objection = max(objection_scores, key=objection_scores.get)
            confidence = objection_scores[best_objection]

            # Получаем рекомендацию
            if best_objection in OBJECTION_CATEGORIES:
                recommendation = OBJECTION_CATEGORIES[best_objection]["recommendation"]
            elif best_objection in self.custom_objections:
                recommendation = self.custom_objections[best_objection].get("recommendation",
                                                                            "Обработать индивидуально")
            else:
                recommendation = "Требует индивидуального подхода"

            return {
                "objection": best_objection,
                "recommendation": recommendation,
                "confidence": confidence,
                "sentiment": sentiment_result
            }

        # Если не нашли конкретное возражение, но есть признаки отказа
        if has_objection:
            # Создаем новую категорию на основе ключевых слов
            new_objection = self._create_new_objection_category(text_lower, sentiment_result)
            if new_objection:
                return new_objection

        return {
            "objection": "Общее возражение",
            "recommendation": "Выяснить конкретную причину возражения",
            "confidence": 1,
            "sentiment": sentiment_result
        }

    def _create_new_objection_category(self, text_lower: str, sentiment_result: Dict) -> Optional[Dict]:
        """Создает новую категорию возражения на основе анализа текста"""
        try:
            # Извлекаем ключевые фразы из текста
            sentences = re.split(r'[.!?]+', text_lower)
            key_phrases = []

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10 and any(word in sentence for word in ["не", "нет", "отказ", "против"]):
                    # Очищаем от служебных слов
                    cleaned = re.sub(r'\b(и|или|но|а|в|на|с|от|по|для|до|при|про|над|под)\b', '', sentence)
                    if len(cleaned.strip()) > 5:
                        key_phrases.append(cleaned.strip()[:50])  # Ограничиваем длину

            if not key_phrases:
                return None

            # Генерируем название новой категории
            new_category_name = f"🔮 Новое возражение {len(self.custom_objections) + 1}"

            # Создаем новую категорию
            self.custom_objections[new_category_name] = {
                "keywords": key_phrases[:5],  # Берем до 5 ключевых фраз
                "recommendation": "Проанализировать и разработать индивидуальный подход",
                "created_date": datetime.datetime.now().isoformat(),
                "sample_text": text_lower[:200],  # Сохраняем образец текста
                "sentiment_info": sentiment_result
            }

            # Сохраняем новые категории
            self._save_custom_objections()

            logger.info(f"Создана новая категория возражения: {new_category_name}")

            return {
                "objection": new_category_name,
                "recommendation": "Проанализировать и разработать индивидуальный подход",
                "confidence": 2,
                "sentiment": sentiment_result,
                "is_new_category": True
            }

        except Exception as e:
            logger.error(f"Ошибка создания новой категории возражения: {e}")
            return None

    def get_all_objection_categories(self) -> Dict:
        """Возвращает все категории возражений (стандартные + пользовательские)"""
        all_categories = OBJECTION_CATEGORIES.copy()
        all_categories.update(self.custom_objections)
        return all_categories

    def extract_key_points(self, text: str) -> List[str]:
        """Извлечение ключевых моментов из текста"""
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

    # Метод для обратной совместимости
    def find_rejection_reason(self, text: str) -> Optional[str]:
        """Устаревший метод для обратной совместимости. Используйте find_objection_reason()"""
        logger.warning("Метод find_rejection_reason устарел. Используйте find_objection_reason()")

        objection_result = self.find_objection_reason(text)
        if objection_result:
            # Возвращаем только название возражения для совместимости
            return objection_result["objection"]
        return None