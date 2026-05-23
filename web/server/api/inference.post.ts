export default defineEventHandler(async (event) => {
  const body = await readBody(event)
  const { action, params } = body

  const backendUrl = process.env.OMNIFF_BACKEND_URL || "http://127.0.0.1:8000"

  try {
    const response = await $fetch(`${backendUrl}/api/${action}`, {
      method: "POST",
      body: params,
      timeout: 120_000,
    })
    return response
  } catch (err: any) {
    throw createError({
      statusCode: err.statusCode || 502,
      message: err.message || "Backend unavailable",
    })
  }
})
