const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "/api"

export function useApi() {
  async function generateText(prompt: string, thinking = "off"): Promise<string> {
    const form = new FormData()
    form.append("prompt", prompt)
    form.append("thinking", thinking)
    const res = await $fetch<{ result: string }>(`${BACKEND_URL}/generate_text`, {
      method: "POST",
      body: form,
    })
    return res.result
  }

  async function translate(text: string, sourceLang: string, targetLang: string): Promise<string> {
    const form = new FormData()
    form.append("text", text)
    form.append("source_lang", sourceLang)
    form.append("target_lang", targetLang)
    const res = await $fetch<{ result: string }>(`${BACKEND_URL}/translate`, {
      method: "POST",
      body: form,
    })
    return res.result
  }

  async function describeImage(file: File, prompt: string): Promise<string> {
    const form = new FormData()
    form.append("file", file)
    form.append("prompt", prompt)
    const res = await $fetch<{ result: string }>(`${BACKEND_URL}/describe`, {
      method: "POST",
      body: form,
    })
    return res.result
  }

  async function transcribeAudio(file: File, language: string): Promise<string> {
    const form = new FormData()
    form.append("file", file)
    form.append("language", language)
    const res = await $fetch<{ result: string }>(`${BACKEND_URL}/transcribe`, {
      method: "POST",
      body: form,
    })
    return res.result
  }

  async function generateImage(prompt: string, seed = -1): Promise<string> {
    const form = new FormData()
    form.append("prompt", prompt)
    form.append("seed", String(seed))
    const res = await fetch(`${BACKEND_URL}/generate_image`, {
      method: "POST",
      body: form,
    })
    const blob = await res.blob()
    return URL.createObjectURL(blob)
  }

  async function generateCode(task: string, language: string): Promise<string> {
    const form = new FormData()
    form.append("task", task)
    form.append("language", language)
    const res = await $fetch<{ result: string }>(`${BACKEND_URL}/generate_code`, {
      method: "POST",
      body: form,
    })
    return res.result
  }

  async function checkHealth(): Promise<Record<string, any>> {
    return await $fetch(`${BACKEND_URL}/health`)
  }

  return { generateText, translate, describeImage, transcribeAudio, generateImage, generateCode, checkHealth }
}
