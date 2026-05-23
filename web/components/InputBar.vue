<script setup lang="ts">
const message = ref("")

function handleSend() {
  if (!message.value.trim()) return
  // TODO: route to workbench processing
  message.value = ""
}

function handleAttach() {
  // TODO: file picker
}
</script>

<template>
  <div class="input-bar">
    <div class="input-bar-inner">
      <button class="input-attach" @click="handleAttach">📎</button>
      <input
        v-model="message"
        type="text"
        class="input-field"
        placeholder="Сообщение, файл, ссылка..."
        @keyup.enter="handleSend"
      />
      <button class="input-mic">🎤</button>
      <button class="input-send" @click="handleSend">▶</button>
    </div>
  </div>
</template>

<style scoped>
.input-bar {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  z-index: 100;
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-top: 1px solid var(--border-light);
  padding: 10px 24px;
}

.input-bar-inner {
  max-width: 860px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  gap: 8px;
}

.input-attach,
.input-mic,
.input-send {
  width: 34px;
  height: 34px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: var(--transition);
  flex-shrink: 0;
  font-size: 16px;
  border: none;
}

.input-attach,
.input-mic {
  background: var(--surface-2);
  border: 1px solid var(--border-light);
}

.input-attach:hover,
.input-mic:hover {
  background: var(--surface-3);
}

.input-send {
  background: var(--accent);
  color: white;
  font-size: 14px;
}

.input-send:hover {
  filter: brightness(1.1);
}

.input-field {
  flex: 1;
  padding: 9px 16px;
  background: var(--surface-2);
  border: 1px solid var(--border-light);
  border-radius: 20px;
  font-size: 14px;
  font-family: inherit;
  color: var(--text);
  outline: none;
  transition: var(--transition);
}

.input-field:focus {
  border-color: var(--accent);
}

.input-field::placeholder {
  color: var(--text-tertiary);
}
</style>
