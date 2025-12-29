const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(detail || `Request failed with ${res.status}`)
  }
  return res.json()
}

export async function fetchValidation({ offset = 0, limit = 50 } = {}) {
  return request(`/validation?offset=${offset}&limit=${limit}`)
}

export async function infer(payload) {
  return request('/infer', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function health() {
  return request('/health')
}

