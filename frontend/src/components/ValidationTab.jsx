import { useEffect, useState } from 'react'
import { fetchValidation } from '../api'

const PAGE_SIZE = 10

const formatTopk = (items) => {
  if (!items || items.length === 0) return '—'
  return items
    .map((item) => `${item.label} (${(item.prob * 100).toFixed(1)}%)`)
    .join(', ')
}

function ValidationTab() {
  const [items, setItems] = useState([])
  const [total, setTotal] = useState(0)
  const [source, setSource] = useState('')
  const [page, setPage] = useState(1) // 1-based
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const loadPage = async (nextPage = page) => {
    const safePage = Math.max(1, Number(nextPage) || 1)
    const nextOffset = (safePage - 1) * PAGE_SIZE
    setLoading(true)
    setError('')
    try {
      const data = await fetchValidation({ offset: nextOffset, limit: PAGE_SIZE })
      setSource(data.source)
      setTotal(data.total)
      setItems(data.items)
      setPage(safePage)
    } catch (err) {
      setError(err.message || 'Failed to load validation results')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadPage(1)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))
  const canPrev = page > 1 && !loading
  const canNext = page < totalPages && !loading

  return (
    <div className="panel">
      <div className="panel-header">
        <div>
          <h2>Validation predictions</h2>
        </div>
        <div className="actions">
          <button onClick={() => loadPage(page)} disabled={loading}>
            {loading ? 'Loading…' : 'Reload'}
          </button>
        </div>
      </div>

      {error && <div className="alert error">{error}</div>}

      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th style={{ width: '10%' }}>Title</th>
              <th style={{ width: '35%' }}>Text</th>
              <th style={{ width: '10%' }}>Product</th>
              <th style={{ width: '7%' }}>Hazard</th>
              <th style={{ width: '13%' }}>Top-k Product</th>
              <th style={{ width: '13%' }}>Top-k Hazard</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item, idx) => (
              <tr key={`${item.title}-${idx}`}>
                <td>
                  <div className="text-cell">{item.title}</div>
                </td>
                <td>
                  <div className="text-cell">{item.text}</div>
                </td>
                <td>
                  <span className="pill blue">{item.pred_product}</span>
                </td>
                <td>
                  <span className="pill purple">{item.pred_hazard}</span>
                </td>
                <td>
                  <div className="text-cell">{formatTopk(item.topk_product)}</div>
                </td>
                <td>
                  <div className="text-cell">{formatTopk(item.topk_hazard)}</div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="panel-footer">
        <span>
          Page {page} / {totalPages} · Showing {items.length} records (total {total})
        </span>
        <div className="actions">
          <button onClick={() => loadPage(page - 1)} disabled={!canPrev}>
            Prev
          </button>
          <button onClick={() => loadPage(page + 1)} disabled={!canNext}>
            Next
          </button>
        </div>
      </div>
    </div>
  )
}

export default ValidationTab

