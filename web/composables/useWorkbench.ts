export interface FileAction {
  id: string
  icon: string
  name: string
  desc: string
  color: string
  optType: "textarea" | "select" | null
  optLabel?: string
  optPlaceholder?: string
  optChoices?: string[]
  route: string
  result: string
  next: string[]
}

export interface FileConfig {
  icon: string
  name: string | null
  meta: string | null
  iconBg: string
  actions: FileAction[]
}

export type FileType = "image" | "video" | "audio" | "document" | "none"

const FILE_CONFIGS: Record<FileType, FileConfig> = {
  image: {
    icon: "🖼",
    name: "landscape.jpg",
    meta: "Изображение · 2.4 MB",
    iconBg: "var(--accent-soft)",
    actions: [
      {
        id: "describe",
        icon: "👁",
        name: "Описать",
        desc: "Что изображено?",
        color: "var(--accent-soft)",
        optType: "textarea",
        optPlaceholder: "Вопрос...",
        route: "Qwen2.5-VL",
        result:
          "Горный пейзаж. Альпийское озеро с бирюзовой водой. Заснеженные вершины в закатном свете. Хвойные деревья по берегу.",
        next: ["🌐 Перевести", "🎨 Стилизовать"],
      },
      {
        id: "ocr",
        icon: "🌐",
        name: "Перевести текст",
        desc: "OCR → перевод",
        color: "var(--green-soft)",
        optType: "select",
        optLabel: "На язык",
        optChoices: ["Русский", "Казахский", "English"],
        route: "VL + Qwen3",
        result: 'Текст: "Mountain Lake Resort"\n→ «Горный Озёрный Курорт»',
        next: ["📋 Копировать", "🔄 Другой язык"],
      },
      {
        id: "upscale",
        icon: "✨",
        name: "Улучшить",
        desc: "×2 разрешение",
        color: "var(--orange-soft)",
        optType: null,
        route: "Real-ESRGAN",
        result: "2400×1600 → 4800×3200 · PNG · 8.7 MB",
        next: ["⬇ Скачать", "🎨 Стилизовать"],
      },
    ],
  },
  video: {
    icon: "🎬",
    name: "lecture.mp4",
    meta: "Видео · 45:12",
    iconBg: "var(--purple-soft)",
    actions: [
      {
        id: "transcribe",
        icon: "📝",
        name: "Транскрибировать",
        desc: "Речь → текст",
        color: "var(--green-soft)",
        optType: "select",
        optLabel: "Язык",
        optChoices: ["Авто", "Қазақша", "Русский"],
        route: "Whisper-turbo",
        result:
          "[00:00] Сәлеметсіздер, дәріске қош келдіңіздер.\n[00:05] Кванттық механика негіздері.\n[00:12] Шрёдингер теңдеуін еске түсірейік.\n\n⏱ 45:12 · Қазақша",
        next: ["🌐 Перевести", "📋 Резюме", "💬 SRT"],
      },
      {
        id: "subtitles",
        icon: "💬",
        name: "Субтитры",
        desc: ".srt файл",
        color: "var(--orange-soft)",
        optType: "select",
        optLabel: "Язык",
        optChoices: ["Оригинал", "Русский", "English"],
        route: "Whisper+Qwen3",
        result:
          "1\n00:00:00 --> 00:00:05\nСәлеметсіздер, дәріске\nқош келдіңіздер.\n\n📄 lecture.srt · 127 строк",
        next: ["⬇ Скачать", "🌐 Перевести"],
      },
    ],
  },
  audio: {
    icon: "🎵",
    name: "interview.m4a",
    meta: "Аудио · 18:34",
    iconBg: "var(--green-soft)",
    actions: [
      {
        id: "transcribe",
        icon: "📝",
        name: "Транскрибировать",
        desc: "Речь → текст",
        color: "var(--green-soft)",
        optType: "select",
        optLabel: "Язык",
        optChoices: ["Авто", "Русский", "Қазақша"],
        route: "Whisper-turbo",
        result:
          "[Спикер 1] Как вы пришли в науку?\n[Спикер 2] Увлёкся физикой после олимпиады.\n\n⏱ 18:34 · 2 спикера",
        next: ["🌐 Перевести", "📋 Резюме"],
      },
      {
        id: "summarize",
        icon: "📋",
        name: "Резюме",
        desc: "Ключевые тезисы",
        color: "var(--orange-soft)",
        optType: null,
        route: "Whisper+Qwen3",
        result:
          "Путь в науку · Журналист + Физик\n• Олимпиада в 9 классе\n• «Физика — язык вселенной»",
        next: ["🌐 Перевести", "⬇ Скачать"],
      },
    ],
  },
  document: {
    icon: "📄",
    name: "paper.pdf",
    meta: "PDF · 14 стр",
    iconBg: "var(--orange-soft)",
    actions: [
      {
        id: "summarize",
        icon: "📋",
        name: "Резюме",
        desc: "Главные тезисы",
        color: "var(--orange-soft)",
        optType: null,
        route: "Qwen3-32B",
        result:
          "KazBERT: 94.2% NER\n+7% vs мульти · Морфемный токенизатор\n14 стр · 47 ссылок",
        next: ["🌐 Перевести", "💬 Спросить"],
      },
      {
        id: "translate",
        icon: "🌐",
        name: "Перевести",
        desc: "Весь документ",
        color: "var(--green-soft)",
        optType: "select",
        optLabel: "На язык",
        optChoices: ["Русский", "English"],
        route: "Qwen3-32B",
        result: "KZ → RU · 14 стр\n📥 paper_RU.pdf",
        next: ["⬇ Скачать", "📋 Резюме"],
      },
      {
        id: "ask",
        icon: "💬",
        name: "Спросить",
        desc: "Вопрос по документу",
        color: "var(--purple-soft)",
        optType: "textarea",
        optPlaceholder: "Вопрос...",
        route: "Qwen3-32B",
        result:
          "KazBERT лучше благодаря морфемному токенизатору и предобучению на 12 ГБ казахского текста.",
        next: ["💬 Ещё вопрос", "📋 Резюме"],
      },
    ],
  },
  none: {
    icon: "✏️",
    name: null,
    meta: null,
    iconBg: "",
    actions: [
      {
        id: "write",
        icon: "📝",
        name: "Написать",
        desc: "Текст, эссе",
        color: "var(--accent-soft)",
        optType: "textarea",
        optPlaceholder: "Тема...",
        route: "Qwen3-32B",
        result:
          "ИИ в образовании Центральной Азии\n\nУниверситеты внедряют AI-инструменты...\n[1200 слов]",
        next: ["🌐 Перевести", "📋 Сократить"],
      },
      {
        id: "generate",
        icon: "🎨",
        name: "Картинка",
        desc: "Текст → изображение",
        color: "var(--purple-soft)",
        optType: "textarea",
        optPlaceholder: "Описание...",
        route: "Z-Image-Turbo",
        result: '🖼 «Горы Алатау» · 1024×1024 · Seed 42',
        next: ["✨ Улучшить", "🔄 Seed"],
      },
      {
        id: "code",
        icon: "💻",
        name: "Код",
        desc: "По описанию",
        color: "var(--green-soft)",
        optType: "textarea",
        optPlaceholder: "Задача...",
        route: "Qwen3-32B",
        result:
          'def analyze(text):\n    words = text.split()\n    return {"count": len(words)}\n\nPython · 3 строки',
        next: ["💬 Объяснить", "🔧 Улучшить"],
      },
      {
        id: "translate",
        icon: "🌐",
        name: "Перевести",
        desc: "На другой язык",
        color: "var(--orange-soft)",
        optType: "textarea",
        optPlaceholder: "Текст...",
        route: "Qwen3-32B",
        result:
          'EN → KZ\n«AI transforms education»\n→ «ЖИ білім беруді өзгертеді»',
        next: ["🔄 Обратный", "📋 Копировать"],
      },
    ],
  },
}

export function useWorkbench() {
  const activeFileType = ref<FileType | null>(null)
  const selectedAction = ref<FileAction | null>(null)
  const chain = ref<string[]>([])
  const resultText = ref("")
  const isProcessing = ref(false)
  const showResult = ref(false)

  const fileConfig = computed(() =>
    activeFileType.value ? FILE_CONFIGS[activeFileType.value] : null,
  )

  function selectFileType(type: FileType) {
    activeFileType.value = type
    selectedAction.value = null
    chain.value = []
    resultText.value = ""
    showResult.value = false
  }

  function selectAction(action: FileAction) {
    selectedAction.value = action
    showResult.value = false
    resultText.value = ""
    if (!action.optType) {
      runAction(action)
    }
  }

  function runAction(action: FileAction) {
    isProcessing.value = true
    showResult.value = true
    resultText.value = ""
    chain.value.push(action.name)

    const lines = action.result.split("\n")
    let i = 0
    const interval = setInterval(() => {
      if (i < lines.length) {
        resultText.value += (i > 0 ? "\n" : "") + lines[i]
        i++
      } else {
        clearInterval(interval)
        isProcessing.value = false
      }
    }, 80)
  }

  function runNextAction(label: string) {
    const name = label.replace(/^[^\s]+\s/, "")
    chain.value.push(name)
    isProcessing.value = true
    resultText.value = `⏳ ${label}...`
    setTimeout(() => {
      resultText.value = `✓ ${label} — готово`
      isProcessing.value = false
    }, 800)
  }

  function removeFile() {
    activeFileType.value = null
    selectedAction.value = null
    chain.value = []
    resultText.value = ""
    showResult.value = false
  }

  return {
    activeFileType,
    selectedAction,
    chain,
    resultText,
    isProcessing,
    showResult,
    fileConfig,
    selectFileType,
    selectAction,
    runAction,
    runNextAction,
    removeFile,
  }
}
