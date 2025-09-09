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
        """Улучшенный поиск возражений с системой двойной проверки"""
        if not text.strip():
            return None

        text_lower = text.lower()

        # Этап 1: Первичная детекция возражения
        has_objection, objection_context = self._detect_objection_presence(text_lower)

        if not has_objection:
            return None

        # Этап 2: Поиск среди стандартных категорий
        standard_objection = self._find_standard_objection(text_lower, objection_context)
        if standard_objection:
            return standard_objection

        # Этап 3: Поиск среди существующих пользовательских категорий
        custom_objection = self._find_custom_objection(text_lower, objection_context)
        if custom_objection:
            return custom_objection

        # Этап 4: Попытка создать новую категорию с жесткой проверкой
        new_objection = self._create_verified_objection(text, text_lower, objection_context)
        if new_objection:
            return new_objection

        # Этап 5: Если ничего не нашли - НЕ заявляем возражение
        logger.info("Обнаружены признаки возражения, но не удалось определить конкретную категорию")
        return None

    def _detect_objection_presence(self, text_lower: str) -> tuple[bool, str]:
        """Детектирует наличие возражения и возвращает контекст"""

        # Сильные индикаторы возражения
        strong_indicators = [
            "не интересно", "не нужно", "не подходит", "не хочу", "не буду",
            "откажусь", "отказываюсь", "против", "не согласен", "не устраивает"
        ]

        # Слабые индикаторы (требуют дополнительного контекста)
        weak_indicators = ["нет", "не", "отказ"]

        # Проверяем сильные индикаторы
        for indicator in strong_indicators:
            if indicator in text_lower:
                # Находим контекст вокруг индикатора
                pos = text_lower.find(indicator)
                context = text_lower[max(0, pos - 50):pos + 100]
                return True, context

        # Проверяем слабые индикаторы с контекстом
        for indicator in weak_indicators:
            if indicator in text_lower:
                pos = text_lower.find(indicator)
                context = text_lower[max(0, pos - 50):pos + 100]

                # Должны быть коммерческие термины рядом
                commercial_terms = [
                    "цена", "стоимость", "деньги", "бюджет", "дорого", "услуга",
                    "продукт", "предложение", "договор", "покупка", "заказ",
                    "время", "сроки", "условия", "требования"
                ]

                if any(term in context for term in commercial_terms):
                    return True, context

        return False, ""

    def _find_standard_objection(self, text_lower: str, context: str) -> Optional[Dict]:
        """Ищет возражение среди стандартных категорий"""

        objection_scores = {}

        for objection_name, objection_data in OBJECTION_CATEGORIES.items():
            score = 0
            keywords = objection_data["keywords"]

            for keyword in keywords:
                # Проверяем в полном тексте
                if keyword in text_lower:
                    score += 2

                # Проверяем в контексте возражения
                if keyword in context:
                    score += 3

            if score >= 3:  # Повышенный порог уверенности
                objection_scores[objection_name] = score

        if objection_scores:
            best_objection = max(objection_scores, key=objection_scores.get)
            recommendation = OBJECTION_CATEGORIES[best_objection]["recommendation"]

            return {
                "objection": best_objection,
                "recommendation": recommendation,
                "confidence": objection_scores[best_objection],
                "sentiment": {"sentiment": "negative", "confidence": 0.8}
            }

        return None

    def _find_custom_objection(self, text_lower: str, context: str) -> Optional[Dict]:
        """Ищет возражение среди пользовательских категорий"""

        objection_scores = {}

        for custom_name, custom_data in self.custom_objections.items():
            score = 0
            keywords = custom_data.get("keywords", [])

            for keyword in keywords:
                if keyword in text_lower:
                    score += 2
                if keyword in context:
                    score += 3

            if score >= 3:
                objection_scores[custom_name] = score

        if objection_scores:
            best_objection = max(objection_scores, key=objection_scores.get)
            recommendation = self.custom_objections[best_objection].get("recommendation", "Индивидуальный подход")

            return {
                "objection": best_objection,
                "recommendation": recommendation,
                "confidence": objection_scores[best_objection],
                "sentiment": {"sentiment": "negative", "confidence": 0.8}
            }

        return None

    def _create_verified_objection(self, original_text: str, text_lower: str, context: str) -> Optional[Dict]:
        """Создает новую категорию возражения с жесткой проверкой качества"""

        # Извлекаем потенциальную суть возражения
        objection_essence = self._extract_verified_essence(text_lower, context)

        if not objection_essence:
            logger.info("Не удалось извлечь четкую суть возражения")
            return None

        # Двойная проверка осмысленности
        if not self._validate_objection_quality(objection_essence, original_text):
            logger.info(f"Возражение '{objection_essence}' не прошло проверку качества")
            return None

        # Проверяем уникальность
        if self._is_duplicate_objection(objection_essence):
            logger.info(f"Возражение '{objection_essence}' является дубликатом")
            return None

        # Создаем новую категорию
        objection_name = f"🔍 {objection_essence.capitalize()}"
        recommendation = self._generate_contextual_recommendation(objection_essence, context)

        self.custom_objections[objection_name] = {
            "keywords": [objection_essence.lower()],
            "recommendation": recommendation,
            "created_date": datetime.datetime.now().isoformat(),
            "source_context": context[:100],
            "validation_passed": True
        }

        self._save_custom_objections()
        logger.info(f"Создана проверенная категория возражения: {objection_name}")

        return {
            "objection": objection_name,
            "recommendation": recommendation,
            "confidence": 3,
            "sentiment": {"sentiment": "negative", "confidence": 0.8},
            "is_new_category": True
        }

    def _extract_verified_essence(self, text_lower: str, context: str) -> Optional[str]:
        """Извлекает проверенную суть возражения"""

        # Паттерны для извлечения четкой сути
        essence_patterns = [
            r'не\s+(подходит|устраивает|интересует|нужен|нужно|требуется)\s+(\w+(?:\s+\w+){0,2})',
            r'слишком\s+(дорого|дорогой|дорогие|высокая|высокие)\s*(\w+(?:\s+\w+){0,1})?',
            r'нет\s+(времени|денег|бюджета|средств|возможности|ресурсов)',
            r'уже\s+(есть|работаем|договорились|заключили)\s+(\w+(?:\s+\w+){0,2})',
            r'(\w+(?:\s+\w+){0,2})\s+не\s+(подходит|устраивает)',
        ]

        for pattern in essence_patterns:
            matches = re.findall(pattern, context.lower())
            for match in matches:
                if isinstance(match, tuple):
                    essence_parts = [part.strip() for part in match if part.strip()]
                    essence = ' '.join(essence_parts)
                else:
                    essence = match.strip()

                # Проверяем минимальное качество
                if len(essence) >= 5 and len(essence.split()) >= 1:
                    return essence[:50]  # Ограничиваем длину

        return None

    def _validate_objection_quality(self, essence: str, full_text: str) -> bool:
        """Проверяет качество извлеченного возражения"""

        # Критерии качества
        checks = [
            len(essence) >= 5,  # Минимальная длина
            len(essence) <= 50,  # Максимальная длина
            not essence.isdigit(),  # Не должно быть только цифрами
            len(essence.split()) >= 1,  # Минимум одно слово
            not all(c in '.,!?-()[]{}' for c in essence),  # Не только знаки препинания
        ]

        # Должны пройти все базовые проверки
        if not all(checks):
            return False

        # Дополнительная проверка: суть должна быть связана с коммерческим контекстом
        commercial_context_words = [
            'цена', 'деньги', 'дорого', 'время', 'услуга', 'продукт',
            'условия', 'договор', 'бюджет', 'стоимость', 'требования',
            'предложение', 'решение', 'покупка', 'заказ'
        ]

        essence_lower = essence.lower()
        full_text_lower = full_text.lower()

        # Либо в самой сути, либо в тексте должны быть коммерческие термины
        has_commercial_context = (
                any(word in essence_lower for word in commercial_context_words) or
                any(word in full_text_lower for word in commercial_context_words)
        )

        return has_commercial_context

    def _is_duplicate_objection(self, essence: str) -> bool:
        """Проверяет, не является ли возражение дубликатом существующих"""

        essence_lower = essence.lower()

        # Проверяем среди стандартных категорий
        for objection_name in OBJECTION_CATEGORIES.keys():
            objection_clean = objection_name.lower()
            if essence_lower in objection_clean or objection_clean in essence_lower:
                return True

        # Проверяем среди пользовательских категорий
        for custom_name, custom_data in self.custom_objections.items():
            custom_clean = custom_name.lower()
            if essence_lower in custom_clean or custom_clean in essence_lower:
                return True

            # Проверяем ключевые слова
            keywords = custom_data.get("keywords", [])
            for keyword in keywords:
                if essence_lower in keyword.lower() or keyword.lower() in essence_lower:
                    return True

        return False

    def _extract_meaningful_objection(self, text_lower: str, objection_indicators: List[str]) -> Optional[Dict]:
        """Извлекает осмысленное возражение из текста или возвращает None"""
        try:
            # Разбиваем на предложения
            sentences = re.split(r'[.!?]+', text_lower)

            # Ищем предложения с возражениями
            objection_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if (len(sentence) > 20 and len(sentence) < 150 and
                        any(indicator in sentence for indicator in objection_indicators)):
                    objection_sentences.append(sentence)

            if not objection_sentences:
                return None

            # Берем самое содержательное предложение
            main_sentence = max(objection_sentences, key=len)

            # Пытаемся извлечь ключевую фразу возражения
            objection_essence = self._extract_objection_essence(main_sentence)

            if not objection_essence or len(objection_essence) < 10:
                return None

            # Проверяем, не существует ли уже такая категория
            objection_name = f"🔍 {objection_essence.capitalize()}"

            # Не создаем, если уже есть похожая
            existing_names = list(OBJECTION_CATEGORIES.keys()) + list(self.custom_objections.keys())
            if any(objection_essence.lower() in existing.lower() for existing in existing_names):
                return None

            # Создаем новую содержательную категорию
            recommendation = self._generate_contextual_recommendation(objection_essence, main_sentence)

            # Сохраняем в пользовательские категории
            self.custom_objections[objection_name] = {
                "keywords": [objection_essence.lower(), main_sentence[:50]],
                "recommendation": recommendation,
                "created_date": datetime.datetime.now().isoformat(),
                "source_sentence": main_sentence
            }

            self._save_custom_objections()
            logger.info(f"Создана содержательная категория возражения: {objection_name}")

            return {
                "objection": objection_name,
                "recommendation": recommendation,
                "confidence": 2,
                "sentiment": {"sentiment": "negative", "confidence": 0.8},
                "is_new_category": True
            }

        except Exception as e:
            logger.error(f"Ошибка извлечения значимого возражения: {e}")
            return None

    def _extract_objection_essence(self, sentence: str) -> str:
        """Извлекает суть возражения из предложения"""

        # Убираем стоп-слова и служебные конструкции
        stop_words = {
            'не', 'нет', 'нас', 'это', 'что', 'как', 'где', 'когда', 'очень', 'так',
            'уже', 'еще', 'только', 'может', 'можно', 'нужно', 'будет', 'есть',
            'у', 'в', 'на', 'с', 'от', 'по', 'для', 'до', 'при', 'про', 'над', 'под'
        }

        # Ищем ключевые фразы после отрицания
        patterns = [
            r'не\s+(\w+(?:\s+\w+){0,2})',  # "не подходит", "не интересует нас"
            r'нет\s+(\w+(?:\s+\w+){0,2})',  # "нет времени", "нет возможности"
            r'слишком\s+(\w+)',  # "слишком дорого"
            r'(\w+)\s+не\s+подходит',  # "цена не подходит"
            r'(\w+)\s+не\s+устраивает',  # "условия не устраивают"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, sentence)
            for match in matches:
                if isinstance(match, tuple):
                    match = ' '.join(match)

                # Очищаем от стоп-слов
                words = match.split()
                clean_words = [w for w in words if w not in stop_words and len(w) > 2]

                if clean_words and len(' '.join(clean_words)) > 5:
                    return ' '.join(clean_words)[:50]

        return ""

    def _generate_contextual_recommendation(self, essence: str, full_sentence: str) -> str:
        """Генерирует рекомендацию на основе контекста возражения"""

        essence_lower = essence.lower()
        sentence_lower = full_sentence.lower()

        # Специфичные рекомендации на основе ключевых слов
        if any(word in essence_lower for word in ["время", "некогда", "занят"]):
            return "Предложить удобное время, ускорить процесс принятия решения"
        elif any(word in essence_lower for word in ["дорого", "цена", "денег", "бюджет"]):
            return "Обосновать ценность предложения, предложить рассрочку или скидку"
        elif any(word in essence_lower for word in ["не нужно", "не требуется", "не актуально"]):
            return "Выявить скрытые потребности, показать дополнительные выгоды"
        elif any(word in essence_lower for word in ["подумать", "решение", "посоветоваться"]):
            return "Предоставить материалы для принятия решения, назначить контрольный звонок"
        elif any(word in essence_lower for word in ["условия", "требования", "не подходит"]):
            return "Выяснить желаемые условия, предложить альтернативные варианты"
        else:
            return f"Проработать возражение '{essence}', предложить индивидуальное решение"

    def _create_new_objection_category(self, text_lower: str, sentiment_result: Dict) -> Optional[Dict]:
        """Создает новую категорию возражения на основе анализа текста с осмысленным названием"""
        try:
            # Извлекаем ключевые фразы из текста
            sentences = re.split(r'[.!?]+', text_lower)
            objection_phrases = []

            # Ищем фразы с отказом и их причины
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 15 and any(word in sentence for word in ["не", "нет", "отказ", "против"]):
                    # Очищаем и сохраняем значимые фразы
                    cleaned = re.sub(r'\b(и|или|но|а|в|на|с|от|по|для|до|при|про|над|под|это|это|что|как|где|когда)\b',
                                     '', sentence)
                    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

                    if len(cleaned) > 10:
                        objection_phrases.append(cleaned)

            if not objection_phrases:
                return None

            # Анализируем основную причину возражения
            main_objection_text = objection_phrases[0][:80] if objection_phrases else "неопределенное возражение"

            # Генерируем осмысленное название на основе содержания
            category_name = self._generate_meaningful_objection_name(main_objection_text, text_lower)

            # Проверяем, не существует ли уже такая категория
            if category_name in self.custom_objections or category_name in OBJECTION_CATEGORIES:
                return None  # Не создаем дубликаты

            # Генерируем рекомендацию на основе содержания
            recommendation = self._generate_objection_recommendation(main_objection_text)

            # Создаем новую категорию
            self.custom_objections[category_name] = {
                "keywords": objection_phrases[:3],  # Берем до 3 ключевых фраз
                "recommendation": recommendation,
                "created_date": datetime.datetime.now().isoformat(),
                "sample_text": main_objection_text,
                "main_phrase": main_objection_text
            }

            # Сохраняем новые категории
            self._save_custom_objections()

            logger.info(f"Создана новая категория возражения: {category_name}")

            return {
                "objection": category_name,
                "recommendation": recommendation,
                "confidence": 2,
                "sentiment": sentiment_result,
                "is_new_category": True
            }

        except Exception as e:
            logger.error(f"Ошибка создания новой категории возражения: {e}")
            return None

    def _generate_meaningful_objection_name(self, main_text: str, full_text: str) -> str:
        """Генерирует осмысленное название возражения на основе содержания"""

        # Паттерны для определения типа возражения
        objection_patterns = {
            "Проблемы с документооборотом": ["документы", "бумаги", "оформление", "договор", "бюрократия"],
            "Недостаток информации": ["не знаю", "не понимаю", "непонятно", "нужно узнать", "информации мало"],
            "Проблемы с внедрением": ["внедрение", "установка", "настройка", "сложно внедрить", "долго настраивать"],
            "Нет необходимых ресурсов": ["нет ресурсов", "нет людей", "нет времени на внедрение", "некому заниматься"],
            "Планы уже на следующий период": ["следующий год", "позже", "в планах", "бюджет следующего года"],
            "Нужно согласование с другими": ["согласовать", "обсудить с командой", "решение не мое", "нужно одобрение"],
            "Сомнения в результате": ["не уверен", "сомневаюсь", "а вдруг не поможет", "не гарантирован результат"],
            "Текущие приоритеты": ["другие приоритеты", "сейчас не до этого", "есть более важные задачи"]
        }

        # Проверяем, подходит ли под существующие паттерны
        for objection_name, keywords in objection_patterns.items():
            if any(keyword in full_text.lower() for keyword in keywords):
                return f"📝 {objection_name}"

        # Если не подошло ни под один паттерн, создаем название на основе ключевых слов
        words = main_text.lower().split()
        key_words = []

        # Отбираем значимые слова
        skip_words = {
            'не', 'нет', 'нас', 'это', 'что', 'как', 'где', 'когда', 'очень', 'так',
            'уже', 'еще', 'только', 'может', 'можно', 'нужно', 'будет', 'есть'
        }

        for word in words:
            if len(word) > 3 and word not in skip_words:
                key_words.append(word)
                if len(key_words) >= 3:
                    break

        if key_words:
            # Создаем краткое описательное название
            key_phrase = ' '.join(key_words[:2])
            return f"❓ Возражение: {key_phrase}"

        return f"❓ Специфичное возражение"

    def _generate_objection_recommendation(self, objection_text: str) -> str:
        """Генерирует рекомендацию на основе содержания возражения"""

        text_lower = objection_text.lower()

        if any(word in text_lower for word in ["документы", "бумаги", "оформление"]):
            return "Упростить документооборот, предложить помощь в оформлении"
        elif any(word in text_lower for word in ["не знаю", "не понимаю", "непонятно"]):
            return "Предоставить детальную информацию и презентацию"
        elif any(word in text_lower for word in ["внедрение", "установка", "настройка"]):
            return "Предложить поддержку при внедрении, обучение команды"
        elif any(word in text_lower for word in ["нет ресурсов", "нет людей", "некому"]):
            return "Предложить выделенную команду или консультационную поддержку"
        elif any(word in text_lower for word in ["следующий", "позже", "планах"]):
            return "Зарезервировать место в планах, предложить пилотный проект"
        elif any(word in text_lower for word in ["согласовать", "обсудить", "одобрение"]):
            return "Помочь с презентацией для руководства, предоставить материалы"
        elif any(word in text_lower for word in ["не уверен", "сомневаюсь", "не гарантирован"]):
            return "Предоставить кейсы, гарантии, возможность тестирования"
        else:
            return "Индивидуальный подход, детальное изучение потребностей"

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