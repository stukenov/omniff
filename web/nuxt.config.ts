export default defineNuxtConfig({
  compatibilityDate: "2025-05-24",
  css: ["~/assets/css/main.css"],
  app: {
    head: {
      title: "OmniFF — FFmpeg для ИИ",
      meta: [
        { charset: "utf-8" },
        { name: "viewport", content: "width=device-width, initial-scale=1" },
      ],
    },
  },
  ssr: false,
})
