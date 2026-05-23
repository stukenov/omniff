<script setup lang="ts">
const historyItems = [
  { icon: "🌐", bg: "var(--green-soft)", name: "Перевод: договор_2024.pdf", meta: "KZ → RU · 3 мин назад" },
  { icon: "🔴", bg: "var(--red-soft)", name: "Live: Лекция по физике (45 мин)", meta: "Транскрипция + Перевод · 12 мин назад" },
  { icon: "🏢", bg: "var(--orange-soft)", name: "Live: Q4 планирование", meta: "6 тезисов · 4 задачи · 1 час назад" },
  { icon: "🎨", bg: "var(--purple-soft)", name: "Генерация: «горы Алатау»", meta: "Seed 42 · 2 часа назад" },
  { icon: "📋", bg: "var(--accent-soft)", name: "Пайплайн: Конспект лекции", meta: "4 шага · 3 часа назад" },
]
</script>

<template>
  <div class="container">
    <div class="page-title">История</div>
    <div class="page-subtitle">Каждая операция сохранена. Переделать — один клик.</div>

    <div class="history-list">
      <div
        v-for="(item, i) in historyItems"
        :key="i"
        class="history-item"
      >
        <div class="history-icon" :style="{ background: item.bg }">
          {{ item.icon }}
        </div>
        <div class="history-info">
          <div class="history-name">{{ item.name }}</div>
          <div class="history-meta">{{ item.meta }}</div>
        </div>
        <button class="history-re">↻</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.page-title { font-size: 22px; font-weight: 700; letter-spacing: -0.03em; margin-bottom: 4px; }
.page-subtitle { font-size: 14px; color: var(--text-secondary); margin-bottom: 20px; }

.history-list { display: flex; flex-direction: column; gap: 6px; }

.history-item {
  display: flex; align-items: center; gap: 12px; padding: 14px 18px;
  background: var(--surface); border: 1px solid var(--border-light);
  border-radius: var(--radius-sm); box-shadow: var(--shadow-sm);
  cursor: pointer; transition: var(--transition);
}
.history-item:hover { border-color: var(--border); box-shadow: var(--shadow-md); }

.history-icon {
  width: 36px; height: 36px; border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; flex-shrink: 0;
}

.history-info { flex: 1; }
.history-name { font-size: 13px; font-weight: 600; margin-bottom: 1px; }
.history-meta { font-size: 11px; color: var(--text-secondary); }

.history-re {
  padding: 5px 12px; font-size: 11px; background: var(--surface-2);
  border: 1px solid var(--border-light); border-radius: 6px;
  color: var(--text-secondary); cursor: pointer; transition: var(--transition);
  font-family: inherit;
}
.history-re:hover { color: var(--accent); border-color: var(--accent); }
</style>
