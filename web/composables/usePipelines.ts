export interface PipeStep {
  name: string
  desc: string
  model: string
  color: string
}

export interface Pipeline {
  cat: string
  emoji: string
  name: string
  desc: string
  time: string
  audience: string
  steps: PipeStep[]
}

export const CATEGORIES: Record<string, string> = {
  edu: "🎓 Образование",
  biz: "🏢 Бизнес",
  content: "🎬 Контент",
  research: "🔬 Исследования",
  personal: "👤 Личное",
}

export const PIPELINES: Pipeline[] = [
  {
    cat: "edu", emoji: "🎓", name: "Конспект лекции",
    desc: "Запись лекции → полный конспект на нужном языке", time: "~3 мин", audience: "Студенты, преподаватели",
    steps: [
      { name: "Транскрипция", desc: "Аудиозапись → текст с таймкодами", model: "Whisper-turbo", color: "var(--green)" },
      { name: "Детекция языка", desc: "Авто-определение языка речи", model: "Whisper", color: "var(--accent)" },
      { name: "Перевод", desc: "KZ↔RU↔EN — на целевой язык", model: "Qwen3-32B", color: "var(--orange)" },
      { name: "Конспект", desc: "Структурированное резюме с тезисами", model: "Qwen3-32B", color: "var(--purple)" },
    ],
  },
  {
    cat: "edu", emoji: "📖", name: "Перевод учебника",
    desc: "PDF учебник → перевод с сохранением структуры", time: "~10 мин", audience: "Студенты, переводчики",
    steps: [
      { name: "OCR + парсинг", desc: "Извлечение текста и структуры документа", model: "Qwen2.5-VL", color: "var(--accent)" },
      { name: "Сегментация", desc: "Разбивка на главы, параграфы, подписи", model: "Qwen3-32B", color: "var(--orange)" },
      { name: "Перевод", desc: "Перевод сегментов с учётом контекста", model: "Qwen3-32B", color: "var(--green)" },
      { name: "Сборка PDF", desc: "Формирование переведённого документа", model: "Системный", color: "var(--purple)" },
    ],
  },
  {
    cat: "edu", emoji: "📝", name: "Подготовка к экзамену",
    desc: "Материалы → флеш-карточки + тест-вопросы", time: "~5 мин", audience: "Студенты",
    steps: [
      { name: "Анализ", desc: "Извлечение ключевых понятий и определений", model: "Qwen3-32B", color: "var(--accent)" },
      { name: "Карточки", desc: "Флеш-карточки: вопрос + ответ", model: "Qwen3-32B", color: "var(--green)" },
      { name: "Тест", desc: "Множественный выбор + открытые вопросы", model: "Qwen3-32B", color: "var(--purple)" },
      { name: "Экспорт", desc: "CSV для Anki или PDF для печати", model: "Системный", color: "var(--orange)" },
    ],
  },
  {
    cat: "biz", emoji: "🏢", name: "Протокол встречи",
    desc: "Запись совещания → протокол + задачи", time: "~4 мин", audience: "Менеджеры, HR",
    steps: [
      { name: "Транскрипция", desc: "Речь с определением спикеров", model: "Whisper-turbo", color: "var(--green)" },
      { name: "Резюме", desc: "Ключевые решения и вопросы", model: "Qwen3-32B", color: "var(--accent)" },
      { name: "Задачи", desc: "Кто, что, когда — автоматически", model: "Qwen3-32B", color: "var(--orange)" },
      { name: "Протокол", desc: "Оформление в готовый документ", model: "Qwen3-32B", color: "var(--purple)" },
    ],
  },
  {
    cat: "biz", emoji: "📑", name: "Анализ договора",
    desc: "Юридический документ → анализ рисков", time: "~3 мин", audience: "Юристы, предприниматели",
    steps: [
      { name: "Парсинг", desc: "Структура договора, ключевые пункты", model: "Qwen3-32B", color: "var(--accent)" },
      { name: "Риски", desc: "Потенциально невыгодные условия", model: "Qwen3-32B", color: "var(--red)" },
      { name: "Сводка", desc: "Стороны, сроки, обязательства", model: "Qwen3-32B", color: "var(--orange)" },
      { name: "Рекомендации", desc: "Вопросы для обсуждения с юристом", model: "Qwen3-32B", color: "var(--green)" },
    ],
  },
  {
    cat: "biz", emoji: "📊", name: "Дайджест отчётов",
    desc: "Несколько PDF → единая сводка с цифрами", time: "~5 мин", audience: "Руководители, аналитики",
    steps: [
      { name: "Загрузка", desc: "Пакетная обработка документов", model: "Системный", color: "var(--accent)" },
      { name: "Извлечение", desc: "Ключевые цифры и выводы", model: "Qwen3-32B", color: "var(--green)" },
      { name: "Сравнение", desc: "Сопоставление между отчётами", model: "Qwen3-32B", color: "var(--orange)" },
      { name: "Дайджест", desc: "Единая сводка с таблицей", model: "Qwen3-32B", color: "var(--purple)" },
    ],
  },
  {
    cat: "content", emoji: "🎬", name: "Субтитры для видео",
    desc: "Видео → субтитры на нескольких языках", time: "~5 мин", audience: "Блогеры, продюсеры",
    steps: [
      { name: "Транскрипция", desc: "Точные таймкоды речи", model: "Whisper-turbo", color: "var(--green)" },
      { name: "Перевод ×3", desc: "Параллельный перевод на 3 языка", model: "Qwen3-32B", color: "var(--accent)" },
      { name: "Форматирование", desc: "Разбивка по 42 символа на строку", model: "Системный", color: "var(--orange)" },
      { name: "Экспорт SRT", desc: "Отдельный .srt для каждого языка", model: "Системный", color: "var(--purple)" },
    ],
  },
  {
    cat: "content", emoji: "🎙", name: "Подкаст → контент",
    desc: "Эпизод → посты, цитаты, описание", time: "~4 мин", audience: "Подкастеры, SMM",
    steps: [
      { name: "Транскрипция", desc: "Текст с разделением спикеров", model: "Whisper-turbo", color: "var(--green)" },
      { name: "Цитаты", desc: "5 ярких цитат для соцсетей", model: "Qwen3-32B", color: "var(--purple)" },
      { name: "Описание", desc: "SEO-оптимизированное + теги", model: "Qwen3-32B", color: "var(--accent)" },
      { name: "Посты", desc: "3 поста из материала эпизода", model: "Qwen3-32B", color: "var(--orange)" },
    ],
  },
  {
    cat: "content", emoji: "📸", name: "Фото → контент",
    desc: "Фотография → описание + посты для соцсетей", time: "~1 мин", audience: "Фотографы, SMM",
    steps: [
      { name: "Описание", desc: "Детальный анализ содержания фото", model: "Qwen2.5-VL", color: "var(--accent)" },
      { name: "Хештеги", desc: "Релевантные теги для Instagram/TikTok", model: "Qwen3-32B", color: "var(--green)" },
      { name: "Посты", desc: "Текст для 3 соцсетей разного формата", model: "Qwen3-32B", color: "var(--purple)" },
    ],
  },
  {
    cat: "research", emoji: "🔬", name: "Обзор статьи",
    desc: "Научная статья → структурированный обзор", time: "~3 мин", audience: "Исследователи, аспиранты",
    steps: [
      { name: "Извлечение", desc: "Abstract, методология, результаты", model: "Qwen3-32B", color: "var(--accent)" },
      { name: "Критика", desc: "Сильные/слабые стороны, ограничения", model: "Qwen3-32B", color: "var(--orange)" },
      { name: "Перевод", desc: "Обзор на нужном языке", model: "Qwen3-32B", color: "var(--green)" },
      { name: "BibTeX", desc: "Ссылка в нужном формате", model: "Системный", color: "var(--purple)" },
    ],
  },
  {
    cat: "research", emoji: "📈", name: "Извлечение данных",
    desc: "Таблицы из PDF/фото → CSV/JSON", time: "~2 мин", audience: "Аналитики, датасаентисты",
    steps: [
      { name: "OCR таблиц", desc: "Распознавание таблиц и графиков", model: "Qwen2.5-VL", color: "var(--accent)" },
      { name: "Структуризация", desc: "Табличный формат с типами", model: "Qwen3-32B", color: "var(--green)" },
      { name: "Валидация", desc: "Проверка целостности данных", model: "Qwen3-32B", color: "var(--orange)" },
      { name: "Экспорт", desc: "CSV / JSON / Excel", model: "Системный", color: "var(--purple)" },
    ],
  },
  {
    cat: "personal", emoji: "🎤", name: "Голосовой дневник",
    desc: "Голосовые заметки → организованный дневник", time: "~2 мин", audience: "Все",
    steps: [
      { name: "Транскрипция", desc: "Голос → текст", model: "Whisper-turbo", color: "var(--green)" },
      { name: "Очистка", desc: "Убрать паразиты, структуризация", model: "Qwen3-32B", color: "var(--accent)" },
      { name: "Теги", desc: "Настроение, тема, ключевые слова", model: "Qwen3-32B", color: "var(--purple)" },
      { name: "Запись", desc: "Добавление в хронологию", model: "Системный", color: "var(--orange)" },
    ],
  },
  {
    cat: "personal", emoji: "🧳", name: "Переводчик путешественника",
    desc: "Фото меню/вывески → перевод + контекст", time: "~30 сек", audience: "Туристы",
    steps: [
      { name: "OCR", desc: "Распознавание текста на фото", model: "Qwen2.5-VL", color: "var(--accent)" },
      { name: "Перевод", desc: "Перевод на ваш язык", model: "Qwen3-32B", color: "var(--green)" },
      { name: "Контекст", desc: "Объяснение блюд, знаков, нюансов", model: "Qwen3-32B", color: "var(--purple)" },
    ],
  },
]

export function usePipelines() {
  const activeCategory = ref("all")
  const expandedIndex = ref(-1)

  const filteredPipelines = computed(() => {
    if (activeCategory.value === "all") return PIPELINES
    return PIPELINES.filter((p) => p.cat === activeCategory.value)
  })

  function setCategory(cat: string) {
    activeCategory.value = cat
    expandedIndex.value = -1
  }

  function toggleExpand(idx: number) {
    expandedIndex.value = expandedIndex.value === idx ? -1 : idx
  }

  return { activeCategory, expandedIndex, filteredPipelines, setCategory, toggleExpand, CATEGORIES }
}
