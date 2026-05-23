<script setup lang="ts">
const { activeCategory, expandedIndex, filteredPipelines, setCategory, toggleExpand, CATEGORIES } =
  usePipelines()

const allCats = computed(() => [
  { id: "all", label: "Все" },
  ...Object.entries(CATEGORIES).map(([id, label]) => ({ id, label })),
])
</script>

<template>
  <div class="container">
    <div class="page-title">Пайплайны</div>
    <div class="page-subtitle">Цепочки операций. Выход → вход следующей.</div>

    <div class="cat-filter">
      <button
        v-for="cat in allCats"
        :key="cat.id"
        class="cat-pill"
        :class="{ active: activeCategory === cat.id }"
        @click="setCategory(cat.id)"
      >
        {{ cat.label }}
      </button>
    </div>

    <div
      v-for="(pipe, idx) in filteredPipelines"
      :key="pipe.name"
      class="pipe-wrapper"
    >
      <div
        class="template-card"
        :class="{ expanded: expandedIndex === idx }"
        @click="toggleExpand(idx)"
      >
        <div class="tc-top">
          <div class="tc-emoji">{{ pipe.emoji }}</div>
          <div class="tc-info">
            <div class="tc-name">{{ pipe.name }}</div>
            <div class="tc-desc">{{ pipe.desc }}</div>
          </div>
        </div>
        <div class="tc-meta">
          <span v-for="step in pipe.steps" :key="step.name" class="tc-pill">
            {{ step.name }}
          </span>
          <span class="tc-time">{{ pipe.time }}</span>
        </div>
      </div>

      <!-- Expanded Preview -->
      <div v-if="expandedIndex === idx" class="pipe-preview">
        <div class="pp-header">
          <div class="pp-emoji">{{ pipe.emoji }}</div>
          <div class="pp-info">
            <div class="pp-title">{{ pipe.name }}</div>
            <div class="pp-desc">{{ pipe.desc }}</div>
          </div>
          <button class="pp-close" @click.stop="toggleExpand(idx)">✕</button>
        </div>
        <div class="pp-meta">
          <div>⏱ {{ pipe.time }}</div>
          <div>📊 {{ pipe.steps.length }} шагов</div>
          <div>👥 {{ pipe.audience }}</div>
        </div>
        <div class="pp-steps">
          <div
            v-for="(step, si) in pipe.steps"
            :key="si"
            class="pp-step"
          >
            <div class="pp-step-rail">
              <div class="pp-step-dot" :style="{ background: step.color }">
                {{ si + 1 }}
              </div>
              <div v-if="si < pipe.steps.length - 1" class="pp-step-line" />
            </div>
            <div class="pp-step-content">
              <div class="pp-step-name">{{ step.name }}</div>
              <div class="pp-step-desc">{{ step.desc }}</div>
              <div class="pp-step-model">{{ step.model }}</div>
            </div>
          </div>
        </div>
        <div class="pp-footer">
          <div class="pp-audience">👥 {{ pipe.audience }}</div>
          <button class="run-btn">▶ Использовать</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.page-title { font-size: 22px; font-weight: 700; letter-spacing: -0.03em; margin-bottom: 4px; }
.page-subtitle { font-size: 14px; color: var(--text-secondary); margin-bottom: 20px; }

.cat-filter { display: flex; gap: 4px; margin-bottom: 20px; flex-wrap: wrap; }
.cat-pill {
  padding: 6px 14px; font-size: 12px; font-weight: 500;
  background: var(--surface); border: 1px solid var(--border-light);
  border-radius: 16px; cursor: pointer; transition: var(--transition);
  color: var(--text-secondary); font-family: inherit;
}
.cat-pill:hover { border-color: var(--accent); color: var(--accent); }
.cat-pill.active { background: var(--accent); color: white; border-color: var(--accent); }

.pipe-wrapper { margin-bottom: 8px; }

.template-card {
  padding: 14px; background: var(--surface); border: 1px solid var(--border-light);
  border-radius: var(--radius); box-shadow: var(--shadow-sm); cursor: pointer; transition: var(--transition);
}
.template-card:hover { border-color: var(--accent); box-shadow: var(--shadow-md); transform: translateY(-1px); }
.template-card.expanded { border-color: var(--accent); box-shadow: 0 0 0 1.5px var(--accent); }

.tc-top { display: flex; align-items: flex-start; gap: 10px; }
.tc-emoji { font-size: 24px; flex-shrink: 0; line-height: 1; }
.tc-info { flex: 1; }
.tc-name { font-size: 14px; font-weight: 600; margin-bottom: 2px; }
.tc-desc { font-size: 12px; color: var(--text-secondary); line-height: 1.4; }
.tc-meta { display: flex; gap: 4px; margin-top: 8px; flex-wrap: wrap; }
.tc-pill { padding: 2px 7px; font-size: 10px; background: var(--surface-2); border-radius: 4px; color: var(--text-secondary); }
.tc-time { padding: 2px 7px; font-size: 10px; background: var(--green-soft); border-radius: 4px; color: var(--green); font-weight: 600; margin-left: auto; }

.pipe-preview {
  margin-top: 8px; background: var(--surface); border: 1px solid var(--border-light);
  border-radius: var(--radius); box-shadow: var(--shadow-md); overflow: hidden;
  animation: slideDown 0.3s var(--transition);
}

.pp-header { display: flex; align-items: center; gap: 10px; padding: 14px 18px; border-bottom: 1px solid var(--border-light); }
.pp-emoji { font-size: 28px; }
.pp-info { flex: 1; }
.pp-title { font-size: 16px; font-weight: 700; }
.pp-desc { font-size: 12px; color: var(--text-secondary); }
.pp-close {
  width: 28px; height: 28px; border-radius: 50%; background: var(--surface-2);
  display: flex; align-items: center; justify-content: center; cursor: pointer;
  font-size: 14px; color: var(--text-secondary); transition: var(--transition); border: none;
}
.pp-close:hover { background: var(--surface-3); color: var(--text); }

.pp-meta { display: flex; gap: 16px; padding: 10px 18px; background: var(--surface-2); border-bottom: 1px solid var(--border-light); font-size: 12px; color: var(--text-secondary); }

.pp-steps { padding: 18px; }
.pp-step { display: flex; gap: 14px; position: relative; padding-bottom: 16px; }
.pp-step:last-child { padding-bottom: 0; }
.pp-step-rail { display: flex; flex-direction: column; align-items: center; width: 24px; flex-shrink: 0; }
.pp-step-dot {
  width: 24px; height: 24px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 700; color: white; flex-shrink: 0; z-index: 1;
}
.pp-step-line { width: 2px; flex: 1; background: var(--border-light); margin-top: 4px; }
.pp-step-content { flex: 1; padding-top: 2px; }
.pp-step-name { font-size: 14px; font-weight: 600; margin-bottom: 2px; }
.pp-step-desc { font-size: 12px; color: var(--text-secondary); line-height: 1.4; }
.pp-step-model { display: inline-block; margin-top: 4px; padding: 2px 8px; font-size: 10px; background: var(--surface-2); border-radius: 4px; color: var(--text-tertiary); }

.pp-footer { padding: 14px 18px; border-top: 1px solid var(--border-light); display: flex; align-items: center; justify-content: space-between; }
.pp-audience { font-size: 12px; color: var(--text-tertiary); }

.run-btn {
  padding: 9px 22px; background: var(--accent); color: white; border: none;
  border-radius: var(--radius-xs); font-size: 13px; font-weight: 600;
  cursor: pointer; transition: var(--transition); font-family: inherit;
}
.run-btn:hover { filter: brightness(1.1); }
</style>
