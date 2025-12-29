import { useState } from 'react'
import ValidationTab from './components/ValidationTab'
import InferenceTab from './components/InferenceTab'
import './App.css'

const tabs = [
  { key: 'validation', label: 'Validation predictions' },
  { key: 'inference', label: 'Live inference' },
]

function App() {
  const [activeTab, setActiveTab] = useState('validation')

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Food Hazard Detection</h1>
          <p className="subtitle">DeBERTa + RoBERTa ensemble demo</p>
        </div>
        <nav className="tabs">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              className={`tab ${activeTab === tab.key ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="app-body">
        {activeTab === 'validation' ? <ValidationTab /> : <InferenceTab />}
      </main>
    </div>
  )
}

export default App
