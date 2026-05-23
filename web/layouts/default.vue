<script setup lang="ts">
const route = useRoute()

const tabs = [
  { id: "workbench", label: "Верстак", path: "/" },
  { id: "live", label: "Live", path: "/live" },
  { id: "pipelines", label: "Пайплайны", path: "/pipelines" },
  { id: "history", label: "История", path: "/history" },
]

const activeTab = computed(() => {
  const match = tabs.find((t) => t.path === route.path)
  return match?.id ?? "workbench"
})
</script>

<template>
  <div>
    <header class="topbar">
      <div class="logo">
        <div class="logo-icon">⚡</div>
        <span class="logo-text">OmniFF</span>
        <span class="logo-sep">/</span>
        <span class="logo-sub">FFmpeg для ИИ</span>
        <div class="status-dot" />
      </div>
      <nav class="nav-tabs">
        <NuxtLink
          v-for="tab in tabs"
          :key="tab.id"
          :to="tab.path"
          class="nav-tab"
          :class="{ active: activeTab === tab.id }"
        >
          {{ tab.label }}
        </NuxtLink>
      </nav>
    </header>

    <slot />

    <InputBar />
  </div>
</template>

<style scoped>
.topbar {
  display: flex;
  align-items: center;
  padding: 12px 24px;
  background: rgba(255, 255, 255, 0.72);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border-light);
  position: sticky;
  top: 0;
  z-index: 100;
}

.logo {
  display: flex;
  align-items: center;
  gap: 10px;
}

.logo-icon {
  width: 30px;
  height: 30px;
  background: linear-gradient(135deg, #0071e3, #5856d6);
  border-radius: 7px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 15px;
  color: white;
}

.logo-text {
  font-size: 17px;
  font-weight: 600;
  letter-spacing: -0.02em;
}

.logo-sep {
  color: var(--border);
  margin: 0 2px;
}

.logo-sub {
  font-size: 13px;
  color: var(--text-secondary);
  font-weight: 400;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-left: 8px;
  background: var(--green);
}

.nav-tabs {
  display: flex;
  gap: 2px;
  margin-left: auto;
  background: var(--surface-2);
  padding: 3px;
  border-radius: 9px;
}

.nav-tab {
  padding: 6px 16px;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-secondary);
  border-radius: 7px;
  cursor: pointer;
  transition: var(--transition);
  user-select: none;
  text-decoration: none;
}

.nav-tab:hover {
  color: var(--text);
}

.nav-tab.active {
  color: var(--text);
  background: var(--surface);
  box-shadow: var(--shadow-sm);
}
</style>
