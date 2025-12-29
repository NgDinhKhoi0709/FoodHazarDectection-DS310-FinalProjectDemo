import { useState } from 'react'
import { infer } from '../api'

const TopKItem = ({ label, prob, color = 'blue' }) => {
  const percentage = (prob * 100).toFixed(1)
  return (
    <div className="topk-item">
      <div className="topk-label">{label}</div>
      <div className="topk-bar-container">
        <div className={`topk-bar ${color}`} style={{ width: `${percentage}%` }} />
      </div>
      <div className="topk-percent">{percentage}%</div>
    </div>
  )
}

function InferenceTab() {
  const [title, setTitle] = useState('')
  const [content, setContent] = useState('')
  const [topK, setTopK] = useState(3)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const payload = { title, content }
      if (topK) payload.top_k = Number(topK)
      const res = await infer(payload)
      setResult(res)
    } catch (err) {
      setError(err.message || 'Inference failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <div>
          <h2>Live inference</h2>
          <p className="muted">Run ensemble prediction on custom title and content.</p>
        </div>
      </div>

      <form className="form" onSubmit={handleSubmit}>
        <label>
          Title
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Enter incident title"
            required
          />
        </label>
        <label>
          Content
          <textarea
            rows={6}
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="Enter incident description"
            required
          />
        </label>
        <label className="inline">
          Top-k
          <input
            type="number"
            min={1}
            max={100}
            value={topK}
            onChange={(e) => setTopK(e.target.value)}
          />
        </label>
        <div className="actions">
          <button type="submit" disabled={loading}>
            {loading ? 'Running…' : 'Run inference'}
          </button>
        </div>
      </form>

      {error && <div className="alert error">{error}</div>}

      {result && (
        <div className="result">
          <h3>Prediction</h3>
          <div className="chips">
            <span className="pill blue">Product: {result.pred_product}</span>
            <span className="pill purple">Hazard: {result.pred_hazard}</span>
          </div>
          <div className="topk">
            <div className="topk-section">
              <h4>Top-k Product</h4>
              <div className="topk-list">
                {result.topk_product.map((item, idx) => (
                  <TopKItem key={idx} label={item.label} prob={item.prob} color="blue" />
                ))}
              </div>
            </div>
            <div className="topk-section">
              <h4>Top-k Hazard</h4>
              <div className="topk-list">
                {result.topk_hazard.map((item, idx) => (
                  <TopKItem key={idx} label={item.label} prob={item.prob} color="purple" />
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default InferenceTab

