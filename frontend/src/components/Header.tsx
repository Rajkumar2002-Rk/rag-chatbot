import { Bot } from 'lucide-react'

export default function Header() {
  return (
    <header className="glass border-b border-gray-100 px-6 py-3.5 flex items-center justify-between sticky top-0 z-10">
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 to-blue-700 flex items-center justify-center shadow-md shadow-blue-200">
          <Bot className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="text-sm font-semibold text-gray-900 tracking-tight">
            AI Document Assistant
          </h1>
          <p className="text-xs text-gray-400">Powered by GPT-4o</p>
        </div>
      </div>

      <div className="flex items-center gap-1.5 text-xs font-medium text-emerald-600 bg-emerald-50 border border-emerald-100 px-3 py-1.5 rounded-full">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
        Online
      </div>
    </header>
  )
}
