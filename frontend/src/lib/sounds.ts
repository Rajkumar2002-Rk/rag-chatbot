const ctx = () => new (window.AudioContext || (window as any).webkitAudioContext)()

function play(freq: number, duration: number, type: OscillatorType = 'sine', volume = 0.15) {
  const ac = ctx()
  const osc = ac.createOscillator()
  const gain = ac.createGain()
  osc.type = type
  osc.frequency.value = freq
  gain.gain.setValueAtTime(volume, ac.currentTime)
  gain.gain.exponentialRampToValueAtTime(0.001, ac.currentTime + duration)
  osc.connect(gain).connect(ac.destination)
  osc.start()
  osc.stop(ac.currentTime + duration)
}

export function playSendSound() {
  play(880, 0.1, 'sine', 0.12)
  setTimeout(() => play(1100, 0.08, 'sine', 0.08), 60)
}

export function playReceiveSound() {
  play(660, 0.12, 'sine', 0.1)
  setTimeout(() => play(880, 0.15, 'sine', 0.12), 80)
}
