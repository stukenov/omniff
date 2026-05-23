<script setup lang="ts">
import type { FileAction, FileType } from "~/composables/useWorkbench"

const {
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
} = useWorkbench()

const quickTypes: { type: FileType; icon: string; label: string }[] = [
  { type: "image", icon: "🖼", label: "Фото" },
  { type: "video", icon: "🎬", label: "Видео" },
  { type: "audio", icon: "🎵", label: "Аудио" },
  { type: "document", icon: "📄", label: "Документ" },
  { type: "none", icon: "✏️", label: "Текст" },
]

function copyResult() {
  navigator.clipboard.writeText(resultText.value)
}
</script>

<template>
  <div class="container">
    <!-- Empty State -->
    <div v-if="!activeFileType" class="empty-state">
      <div class="empty-title">Любой вход. Любой выход.</div>
      <div class="empty-subtitle">
        Бросьте файл, вставьте из буфера или начните печатать
      </div>
      <div class="quick-types">
        <button
          v-for="qt in quickTypes"
          :key="qt.type"
          class="quick-type"
          @click="selectFileType(qt.type)"
        >
          {{ qt.icon }} {{ qt.label }}
        </button>
      </div>
    </div>

    <!-- Active File -->
    <div
      v-if="activeFileType && activeFileType !== 'none' && fileConfig"
      class="active-file"
    >
      <div class="af-icon" :style="{ background: fileConfig.iconBg }">
        {{ fileConfig.icon }}
      </div>
      <div class="af-info">
        <div class="af-name">{{ fileConfig.name }}</div>
        <div class="af-meta">{{ fileConfig.meta }}</div>
      </div>
      <button class="af-remove" @click="removeFile">✕</button>
    </div>

    <!-- Breadcrumbs -->
    <div v-if="chain.length > 1" class="breadcrumbs">
      <template v-for="(step, i) in chain" :key="i">
        <span v-if="i > 0" class="bc-arrow">→</span>
        <span class="bc-item">{{ step }}</span>
      </template>
    </div>

    <!-- Actions Grid -->
    <div v-if="activeFileType && fileConfig" class="actions-section">
      <div class="actions-label">
        {{ activeFileType === "none" ? "Что создать?" : "Что сделать?" }}
      </div>
      <div class="actions-grid">
        <button
          v-for="action in fileConfig.actions"
          :key="action.id"
          class="action-btn"
          :class="{ selected: selectedAction?.id === action.id }"
          @click="selectAction(action)"
        >
          <div class="action-icon" :style="{ background: action.color }">
            {{ action.icon }}
          </div>
          <div>
            <div class="action-name">{{ action.name }}</div>
            <div class="action-desc">{{ action.desc }}</div>
          </div>
        </button>
      </div>
    </div>

    <!-- Options Panel -->
    <div
      v-if="selectedAction?.optType"
      class="options-panel"
    >
      <div v-if="selectedAction.optType === 'textarea'" class="options-col">
        <textarea
          class="option-textarea"
          :placeholder="selectedAction.optPlaceholder"
        />
        <button class="run-btn" @click="runAction(selectedAction!)">
          ▶ {{ selectedAction.name }}
        </button>
      </div>
      <div v-else-if="selectedAction.optType === 'select'" class="options-row">
        <div class="option-group">
          <div class="option-label">{{ selectedAction.optLabel }}</div>
          <select class="option-select">
            <option
              v-for="choice in selectedAction.optChoices"
              :key="choice"
            >
              {{ choice }}
            </option>
          </select>
        </div>
        <button class="run-btn" @click="runAction(selectedAction!)">
          ▶ {{ selectedAction.name }}
        </button>
      </div>
    </div>

    <!-- Result Area -->
    <div v-if="showResult" class="result-area">
      <div class="result-box">
        <div class="result-header">
          <div class="result-title">
            <span class="dot" />
            <span>{{ selectedAction?.name }}</span>
            <span class="result-route">{{ selectedAction?.route }}</span>
          </div>
          <div class="result-actions">
            <button class="result-action-btn" @click="copyResult">📋</button>
            <button class="result-action-btn">⬇</button>
          </div>
        </div>
        <div class="result-content">{{ resultText }}</div>
        <div
          v-if="!isProcessing && selectedAction?.next?.length"
          class="next-actions"
        >
          <div class="next-label">Что дальше?</div>
          <div class="next-pills">
            <button
              v-for="nxt in selectedAction!.next"
              :key="nxt"
              class="next-pill"
              @click="runNextAction(nxt)"
            >
              {{ nxt }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.empty-state {
  text-align: center;
  padding: 50px 20px 30px;
}

.empty-title {
  font-size: 26px;
  font-weight: 700;
  letter-spacing: -0.03em;
  margin-bottom: 6px;
}

.empty-subtitle {
  font-size: 14px;
  color: var(--text-secondary);
  margin-bottom: 28px;
}

.quick-types {
  display: flex;
  gap: 8px;
  justify-content: center;
  flex-wrap: wrap;
}

.quick-type {
  padding: 8px 18px;
  font-size: 13px;
  font-weight: 500;
  background: var(--surface);
  border: 1px solid var(--border-light);
  border-radius: 20px;
  cursor: pointer;
  transition: var(--transition);
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  gap: 6px;
  box-shadow: var(--shadow-sm);
  font-family: inherit;
}

.quick-type:hover {
  border-color: var(--accent);
  color: var(--accent);
  background: var(--accent-soft);
}

.active-file {
  display: flex;
  background: var(--surface);
  border: 1px solid var(--border-light);
  border-radius: var(--radius);
  padding: 14px 18px;
  box-shadow: var(--shadow-sm);
  align-items: center;
  gap: 12px;
  margin-bottom: 14px;
}

.af-icon {
  width: 40px;
  height: 40px;
  border-radius: var(--radius-xs);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  flex-shrink: 0;
}

.af-info { flex: 1; }
.af-name { font-size: 14px; font-weight: 600; }
.af-meta { font-size: 12px; color: var(--text-secondary); }

.af-remove {
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: var(--red-soft);
  color: var(--red);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-size: 12px;
  font-weight: 700;
  border: none;
}

.breadcrumbs {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 12px;
  font-size: 12px;
  flex-wrap: wrap;
}

.bc-item {
  padding: 3px 10px;
  background: var(--accent-soft);
  color: var(--accent);
  border-radius: 4px;
  font-weight: 500;
}

.bc-arrow { color: var(--border); }

.actions-section { margin-bottom: 14px; }

.actions-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-tertiary);
  margin-bottom: 8px;
}

.actions-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 6px;
}

.action-btn {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 14px;
  background: var(--surface);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: var(--transition);
  box-shadow: var(--shadow-sm);
  font-family: inherit;
  text-align: left;
}

.action-btn:hover {
  border-color: var(--accent);
  background: var(--accent-soft);
}

.action-btn.selected {
  border-color: var(--accent);
  background: var(--accent-soft);
  box-shadow: 0 0 0 1.5px var(--accent);
}

.action-icon {
  width: 32px;
  height: 32px;
  border-radius: 7px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  flex-shrink: 0;
}

.action-name { font-size: 13px; font-weight: 600; }
.action-desc { font-size: 11px; color: var(--text-secondary); }

.options-panel {
  margin-bottom: 14px;
  padding: 16px;
  background: var(--surface);
  border: 1px solid var(--border-light);
  border-radius: var(--radius);
  box-shadow: var(--shadow-sm);
}

.options-row {
  display: flex;
  gap: 10px;
  align-items: flex-end;
  flex-wrap: wrap;
}

.options-col {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.option-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
  min-width: 130px;
}

.option-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-secondary);
  font-weight: 600;
}

.option-select,
.option-textarea {
  padding: 9px 12px;
  background: var(--surface-2);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-xs);
  color: var(--text);
  font-size: 13px;
  font-family: inherit;
  outline: none;
  transition: var(--transition);
}

.option-select:focus,
.option-textarea:focus {
  border-color: var(--accent);
}

.option-textarea {
  min-height: 70px;
  resize: vertical;
  width: 100%;
}

.run-btn {
  padding: 9px 22px;
  background: var(--accent);
  color: white;
  border: none;
  border-radius: var(--radius-xs);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: var(--transition);
  font-family: inherit;
  white-space: nowrap;
}

.run-btn:hover { filter: brightness(1.1); }
.run-btn:active { transform: scale(0.97); }

.result-area { margin-bottom: 16px; }

.result-box {
  background: var(--surface);
  border: 1px solid var(--border-light);
  border-radius: var(--radius);
  box-shadow: var(--shadow-md);
  overflow: hidden;
}

.result-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  border-bottom: 1px solid var(--border-light);
  background: var(--surface-2);
}

.result-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  gap: 6px;
}

.result-title .dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--green);
}

.result-route {
  font-size: 10px;
  color: var(--text-tertiary);
  padding: 2px 6px;
  background: var(--surface);
  border-radius: 4px;
  margin-left: 6px;
}

.result-actions {
  display: flex;
  gap: 4px;
}

.result-action-btn {
  padding: 4px 10px;
  font-size: 11px;
  background: var(--surface);
  border: 1px solid var(--border-light);
  border-radius: 6px;
  color: var(--text-secondary);
  cursor: pointer;
  transition: var(--transition);
  font-family: inherit;
}

.result-action-btn:hover {
  color: var(--text);
  border-color: var(--border);
}

.result-content {
  padding: 16px;
  font-size: 14px;
  line-height: 1.75;
  white-space: pre-wrap;
  min-height: 50px;
}

.next-actions {
  padding: 12px 16px;
  border-top: 1px solid var(--border-light);
  background: var(--surface-2);
}

.next-label {
  font-size: 11px;
  color: var(--text-tertiary);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 6px;
}

.next-pills {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

.next-pill {
  padding: 5px 12px;
  font-size: 12px;
  font-weight: 500;
  background: var(--surface);
  border: 1px solid var(--border-light);
  border-radius: 16px;
  cursor: pointer;
  transition: var(--transition);
  color: var(--text-secondary);
  font-family: inherit;
}

.next-pill:hover {
  border-color: var(--accent);
  color: var(--accent);
  background: var(--accent-soft);
}
</style>
