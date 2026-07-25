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
    name: null,
    meta: null,
    iconBg: "var(--accent-soft)",
    actions: [
      {
        id: "describe",
        icon: "👁",
        name: "Описать",
        desc: "Что изображено?",
        color: "var(--accent-soft)",
        optType: "textarea",
        optPlaceholder: "Вопрос к изображению...",
        route: "Qwen2.5-VL",
        next: ["🌐 Перевести"],
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
        next: ["📋 Копировать"],
      },
    ],
  },
  video: {
    icon: "🎬",
    name: null,
    meta: null,
    iconBg: "var(--purple-soft)",
    actions: [
      {
        id: "describe",
        icon: "👁",
        name: "Описать",
        desc: "Что на видео?",
        color: "var(--green-soft)",
        optType: "textarea",
        optPlaceholder: "Вопрос к видео...",
        route: "Qwen2.5-VL",
        next: ["🌐 Перевести"],
      },
    ],
  },
  audio: {
    icon: "🎵",
    name: null,
    meta: null,
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
        optChoices: ["Авто", "Русский", "English", "Қазақша"],
        route: "Whisper-turbo",
        next: ["🌐 Перевести", "📋 Резюме"],
      },
    ],
  },
  document: {
    icon: "📄",
    name: null,
    meta: null,
    iconBg: "var(--orange-soft)",
    actions: [
      {
        id: "summarize",
        icon: "📋",
        name: "Резюме",
        desc: "Главные тезисы",
        color: "var(--orange-soft)",
        optType: null,
        route: "Qwen3-8B",
        next: ["🌐 Перевести"],
      },
      {
        id: "ask",
        icon: "💬",
        name: "Спросить",
        desc: "Вопрос по документу",
        color: "var(--purple-soft)",
        optType: "textarea",
        optPlaceholder: "Вопрос...",
        route: "Qwen3-8B",
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
        route: "Qwen3-8B",
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
        next: ["🔄 Другой Seed"],
      },
      {
        id: "code",
        icon: "💻",
        name: "Код",
        desc: "По описанию",
        color: "var(--green-soft)",
        optType: "textarea",
        optPlaceholder: "Задача...",
        route: "Qwen3-8B",
        next: ["💬 Объяснить"],
      },
      {
        id: "translate",
        icon: "🌐",
        name: "Перевести",
        desc: "На другой язык",
        color: "var(--orange-soft)",
        optType: "textarea",
        optPlaceholder: "Текст...",
        route: "Qwen3-8B",
        next: ["🔄 Обратный"],
      },
    ],
  },
}

const LANG_MAP: Record<string, string> = {
  "Авто": "",
  "Русский": "ru",
  "English": "en",
  "Қазақша": "kk",
}

export function useWorkbench() {
  const activeFileType = ref<FileType | null>(null)
  const selectedAction = ref<FileAction | null>(null)
  const chain = ref<string[]>([])
  const resultText = ref("")
  const resultImage = ref<string | null>(null)
  const isProcessing = ref(false)
  const showResult = ref(false)
  const optionValue = ref("")
  const uploadedFile = ref<File | null>(null)

  const fileConfig = computed(() =>
    activeFileType.value ? FILE_CONFIGS[activeFileType.value] : null,
  )

  function selectFileType(type: FileType) {
    activeFileType.value = type
    selectedAction.value = null
    chain.value = []
    resultText.value = ""
    resultImage.value = null
    showResult.value = false
    optionValue.value = ""
  }

  function selectAction(action: FileAction) {
    selectedAction.value = action
    showResult.value = false
    resultText.value = ""
    resultImage.value = null
    optionValue.value = action.optChoices?.[0] ?? ""
    if (!action.optType) {
      runAction(action)
    }
  }

  function setFile(file: File) {
    uploadedFile.value = file
    const ext = file.name.split(".").pop()?.toLowerCase() ?? ""
    const imgExts = ["jpg", "jpeg", "png", "gif", "webp", "bmp"]
    const vidExts = ["mp4", "mov", "avi", "mkv", "webm"]
    const audExts = ["mp3", "wav", "flac", "ogg", "m4a", "aac"]
    if (imgExts.includes(ext)) selectFileType("image")
    else if (vidExts.includes(ext)) selectFileType("video")
    else if (audExts.includes(ext)) selectFileType("audio")
    else selectFileType("document")
    const cfg = FILE_CONFIGS[activeFileType.value!]
    cfg.name = file.name
    cfg.meta = `${(file.size / 1024 / 1024).toFixed(1)} MB`
  }

  async function runAction(action: FileAction) {
    const api = useApi()
    isProcessing.value = true
    showResult.value = true
    resultText.value = "⏳ Обработка..."
    resultImage.value = null
    chain.value.push(action.name)

    try {
      let result = ""

      switch (action.id) {
        case "write":
          result = await api.generateText(optionValue.value || "Напиши текст")
          break
        case "generate": {
          const url = await api.generateImage(optionValue.value || "abstract art")
          resultImage.value = url
          result = `Изображение сгенерировано`
          break
        }
        case "code":
          result = await api.generateCode(optionValue.value || "hello world", "Python")
          break
        case "translate":
          result = await api.translate(optionValue.value || "", "", "Русский")
          break
        case "describe":
          if (uploadedFile.value) {
            result = await api.describeImage(uploadedFile.value, optionValue.value || "Опишите изображение подробно")
          } else {
            result = "Загрузите файл для описания"
          }
          break
        case "ocr":
          if (uploadedFile.value) {
            const ocrText = await api.describeImage(uploadedFile.value, "Read ALL text visible in this image exactly as written.")
            const lang = optionValue.value || "Русский"
            result = await api.translate(ocrText, "", lang)
            result = `Распознанный текст:\n${ocrText}\n\nПеревод (${lang}):\n${result}`
          } else {
            result = "Загрузите изображение с текстом"
          }
          break
        case "transcribe":
          if (uploadedFile.value) {
            const lang = LANG_MAP[optionValue.value] ?? ""
            result = await api.transcribeAudio(uploadedFile.value, lang)
          } else {
            result = "Загрузите аудио файл"
          }
          break
        case "summarize":
          if (uploadedFile.value) {
            const form = new FormData()
            form.append("file", uploadedFile.value)
            form.append("prompt", "Кратко изложите этот документ")
            const res = await $fetch<{ result: string }>("/api/summarize", { method: "POST", body: form })
            result = res.result
          } else {
            result = "Загрузите документ"
          }
          break
        case "ask":
          if (uploadedFile.value) {
            const form = new FormData()
            form.append("file", uploadedFile.value)
            form.append("prompt", optionValue.value || "О чём этот документ?")
            const res = await $fetch<{ result: string }>("/api/summarize", { method: "POST", body: form })
            result = res.result
          } else {
            result = "Загрузите документ"
          }
          break
        default:
          result = await api.generateText(optionValue.value || action.name)
      }

      resultText.value = result
    } catch (e: any) {
      resultText.value = `Ошибка: ${e?.data?.message || e?.message || "Не удалось выполнить"}`
    } finally {
      isProcessing.value = false
    }
  }

  async function runNextAction(label: string) {
    const name = label.replace(/^[^\s]+\s/, "")
    chain.value.push(name)
    isProcessing.value = true
    const api = useApi()

    try {
      if (name === "Перевести") {
        resultText.value = await api.translate(resultText.value, "", "Русский")
      } else if (name === "Сократить" || name === "Резюме") {
        resultText.value = await api.generateText(`Кратко изложи:\n\n${resultText.value}`)
      } else if (name === "Объяснить") {
        resultText.value = await api.generateText(`Объясни этот код:\n\n${resultText.value}`)
      } else if (name === "Обратный") {
        resultText.value = await api.translate(resultText.value, "", "English")
      } else if (name === "Другой Seed") {
        const url = await api.generateImage(optionValue.value || "abstract art")
        resultImage.value = url
        resultText.value = "Новое изображение сгенерировано"
      } else {
        resultText.value = `✓ ${label} — готово`
      }
    } catch (e: any) {
      resultText.value = `Ошибка: ${e?.message || "Не удалось"}`
    } finally {
      isProcessing.value = false
    }
  }

  function removeFile() {
    activeFileType.value = null
    selectedAction.value = null
    chain.value = []
    resultText.value = ""
    resultImage.value = null
    showResult.value = false
    uploadedFile.value = null
  }

  return {
    activeFileType,
    selectedAction,
    chain,
    resultText,
    resultImage,
    isProcessing,
    showResult,
    optionValue,
    uploadedFile,
    fileConfig,
    selectFileType,
    selectAction,
    setFile,
    runAction,
    runNextAction,
    removeFile,
  }
}
