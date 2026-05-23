<script setup lang="ts">
interface LiveMode {
  id: string
  icon: string
  label: string
  desc: string
  lang: string
  showLegend: boolean
}

const modes: LiveMode[] = [
  { id: "transcribe", icon: "📝", label: "Транскрипция", desc: "Речь → текст в реальном времени", lang: "Қазақша", showLegend: false },
  { id: "translate", icon: "🌐", label: "Перевод", desc: "Речь на одном языке → текст + перевод на другой", lang: "KZ → RU", showLegend: false },
  { id: "meeting", icon: "🏢", label: "Встреча", desc: "Авто-заметки: тезисы, решения, задачи", lang: "Русский", showLegend: true },
  { id: "describe", icon: "👁", label: "Описание", desc: "Камера/экран → непрерывное описание происходящего", lang: "Русский", showLegend: false },
  { id: "dictation", icon: "✍️", label: "Диктовка", desc: "Голос → форматированный текст с пунктуацией и абзацами", lang: "Русский", showLegend: false },
]

const currentMode = ref("transcribe")
const isRecording = ref(false)
const isPaused = ref(false)
const seconds = ref(0)
let timerHandle: ReturnType<typeof setInterval> | null = null

const activeMode = computed(() => modes.find((m) => m.id === currentMode.value)!)

const timerText = computed(() => {
  const m = Math.floor(seconds.value / 60)
  const s = seconds.value % 60
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
})

function selectMode(id: string) {
  stopRecording()
  currentMode.value = id
}

function startRecording() {
  isRecording.value = true
  isPaused.value = false
  seconds.value = 0
  timerHandle = setInterval(() => {
    if (!isPaused.value) seconds.value++
  }, 1000)
}

function togglePause() {
  isPaused.value = !isPaused.value
}

function stopRecording() {
  isRecording.value = false
  isPaused.value = false
  seconds.value = 0
  if (timerHandle) {
    clearInterval(timerHandle)
    timerHandle = null
  }
}

onUnmounted(() => {
  if (timerHandle) clearInterval(timerHandle)
})
</script>

<template>
  <div class="container">
    <div style="text-align: center; padding: 16px 0 8px">
      <div class="page-title">Live</div>
      <div class="page-subtitle">Обработка в реальном времени</div>
    </div>

    <div class="live-modes">
      <button
        v-for="mode in modes"
        :key="mode.id"
        class="live-mode-btn"
        :class="{ active: currentMode === mode.id }"
        @click="selectMode(mode.id)"
      >
        <div class="lm-icon">{{ mode.icon }}</div>
        <div class="lm-label">{{ mode.label }}</div>
      </button>
    </div>

    <div class="live-mode-desc">{{ activeMode.desc }}</div>

    <div class="live-surface">
      <div class="live-status">
        <div class="live-rec">
          <div class="live-rec-dot" :class="{ active: isRecording && !isPaused }" />
          <span class="live-rec-label" :class="{ active: isRecording }">LIVE</span>
        </div>
        <div class="live-lang-pill">{{ activeMode.lang }}</div>
        <div class="live-timer">{{ timerText }}</div>
      </div>

      <div class="live-content">
        <div v-if="!isRecording" class="live-placeholder">
          Нажмите «Запись» для начала
        </div>
        <div v-else class="live-placeholder">
          Идёт запись... (демо)
        </div>
      </div>

      <div class="live-controls">
        <button
          class="live-btn rec"
          :disabled="isRecording"
          @click="startRecording"
        >
          ● Запись
        </button>
        <button class="live-btn" @click="togglePause">
          {{ isPaused ? "▶ Далее" : "⏸ Пауза" }}
        </button>
        <button class="live-btn stop" @click="stopRecording">■ Стоп</button>
        <button class="live-btn live-export">⬇ Экспорт</button>
      </div>
    </div>

    <div v-if="activeMode.showLegend" class="live-legend">
      <div class="live-legend-item">
        <div class="live-legend-dot" style="background: var(--green)" /> Тезис
      </div>
      <div class="live-legend-item">
        <div class="live-legend-dot" style="background: var(--orange)" /> Решение
      </div>
      <div class="live-legend-item">
        <div class="live-legend-dot" style="background: var(--accent)" /> Задача
      </div>
    </div>
  </div>
</template>

<style scoped>
.page-title { font-size: 26px; font-weight: 700; letter-spacing: -0.03em; margin-bottom: 6px; }
.page-subtitle { font-size: 14px; color: var(--text-secondary); margin-bottom: 16px; }

.live-modes {
  display: flex; gap: 4px; justify-content: center; margin-bottom: 20px;
  background: var(--surface-2); padding: 4px; border-radius: var(--radius-sm);
  max-width: 560px; margin-left: auto; margin-right: auto;
}

.live-mode-btn {
  flex: 1; padding: 10px 6px; text-align: center; border-radius: var(--radius-xs);
  cursor: pointer; transition: var(--transition); user-select: none; border: none;
  background: transparent; font-family: inherit;
}
.live-mode-btn:hover { background: rgba(255, 255, 255, 0.5); }
.live-mode-btn.active { background: var(--surface); box-shadow: var(--shadow-sm); }

.lm-icon { font-size: 20px; margin-bottom: 2px; }
.lm-label { font-size: 10px; font-weight: 600; color: var(--text-secondary); }
.live-mode-btn.active .lm-label { color: var(--text); }

.live-mode-desc { text-align: center; font-size: 13px; color: var(--text-tertiary); margin-bottom: 16px; }

.live-surface {
  background: var(--surface); border: 1px solid var(--border-light);
  border-radius: var(--radius); box-shadow: var(--shadow-md); overflow: hidden;
}

.live-status {
  display: flex; align-items: center; gap: 10px; padding: 10px 18px;
  border-bottom: 1px solid var(--border-light);
}

.live-rec { display: flex; align-items: center; gap: 6px; }
.live-rec-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--red); opacity: 0.3; }
.live-rec-dot.active { opacity: 1; animation: blink-dot 1s infinite; }
.live-rec-label { font-size: 12px; font-weight: 700; color: var(--red); opacity: 0.4; }
.live-rec-label.active { opacity: 1; }

.live-timer {
  font-size: 13px; font-weight: 600; color: var(--text-secondary);
  font-variant-numeric: tabular-nums; margin-left: auto;
}

.live-lang-pill {
  padding: 4px 10px; font-size: 11px; background: var(--surface-2);
  border: 1px solid var(--border-light); border-radius: 12px;
  color: var(--text-secondary); font-weight: 500;
}

.live-content { padding: 20px; min-height: 220px; }
.live-placeholder { color: var(--text-tertiary); text-align: center; padding-top: 80px; font-size: 14px; }

.live-controls {
  display: flex; gap: 8px; padding: 12px 18px; border-top: 1px solid var(--border-light);
  background: var(--surface-2); align-items: center;
}

.live-btn {
  padding: 7px 16px; font-size: 12px; font-weight: 600;
  background: var(--surface); border: 1px solid var(--border-light);
  border-radius: var(--radius-xs); cursor: pointer; transition: var(--transition);
  font-family: inherit; color: var(--text-secondary);
}
.live-btn:hover { border-color: var(--accent); color: var(--accent); }
.live-btn.rec { background: var(--red); color: white; border-color: var(--red); }
.live-btn.rec:hover { filter: brightness(1.1); }
.live-btn.rec:disabled { opacity: 0.5; cursor: default; filter: none; }
.live-btn.stop { color: var(--red); border-color: var(--red); background: var(--red-soft); }
.live-export { margin-left: auto; }

.live-legend { display: flex; gap: 16px; justify-content: center; margin-top: 12px; font-size: 11px; color: var(--text-tertiary); }
.live-legend-item { display: flex; align-items: center; gap: 4px; }
.live-legend-dot { width: 6px; height: 6px; border-radius: 50%; }
</style>
