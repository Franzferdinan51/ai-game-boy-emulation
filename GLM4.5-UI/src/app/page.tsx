'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import Header from '@/components/Header'
import AIPanel from '@/components/AIPanel'
import GamePanel from '@/components/GamePanel'
import StatePanel from '@/components/StatePanel'
import SettingsModal from '@/components/SettingsModal'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import useInterval from '@/hooks/useInterval'
import type { 
  Achievement, 
  Character, 
  Objective, 
  Item, 
  GameAction, 
  ChatMessage, 
  AppSettings, 
  AiModel, 
  PlayerStats, 
  MapData 
} from '@/types/assistant-ui'
import { AIState } from '@/types/assistant-ui'
import { fetchAvailableModels, getAIThoughtAndAction, getChatResponse } from '@/services/geminiService'
import { getScreen, sendAction, getCharacters, getGameState, loadRom, saveState as saveGameState, loadState as loadGameState } from '@/services/backendService'

const SAVE_KEY = 'ai-game-assistant-save-v1'
const STREAM_INTERVAL_MS = 250
const IDLE_STATE_REFRESH_MS = 2000

export default function Home() {
  const [aiState, setAiState] = useState<AIState>(AIState.IDLE)
  const [gameScreenUrl, setGameScreenUrl] = useState<string | null>(null)
  const [actionHistory, setActionHistory] = useState<GameAction[]>([])
  const [isRomLoaded, setIsRomLoaded] = useState(false)
  
  // Dynamic Game State
  const [characters, setCharacters] = useState<Character[]>([])
  const [mapInfo, setMapInfo] = useState<MapData>({ name: 'Loading...', coords: [0, 0], pointsOfInterest: []})
  const [achievements, setAchievements] = useState<Achievement[]>([])
  const [objectives, setObjectives] = useState<Objective[]>([
    { id: 1, text: 'Complete the first level', completed: false },
    { id: 2, text: 'Find the hidden bonus area', completed: false },
    { id: 3, text: 'Achieve a high score', completed: false },
  ])
  const [inventory, setInventory] = useState<Item[]>([])
  const [playerStats, setPlayerStats] = useState<PlayerStats>({ runtime: '00:00:00', steps: 0, score: 0 })
  const [dialogue, setDialogue] = useState('')
  
  // State for AIPanel
  const [aiGoal, setAiGoal] = useState<string>('Complete the game with the highest score possible.')
  const [currentReasoning, setCurrentReasoning] = useState<string>('Please load a ROM to begin.')
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([])
  const [chatInput, setChatInput] = useState('')
  const [isChatting, setIsChatting] = useState(false)
  
  // State for Settings
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [appSettings, setAppSettings] = useState<AppSettings>({
    aiActionInterval: 4000,
    backendUrl: 'http://localhost:5000',
    aiProvider: 'google',
    googleApiKey: '',
    openrouterApiKey: '',
    lmStudioUrl: 'http://localhost:1234',
    selectedModel: '',
  })
  const [availableModels, setAvailableModels] = useState<AiModel[]>([])
  const [isModelListLoading, setIsModelListLoading] = useState(false)
  const [saveStatus, setSaveStatus] = useState<'saving' | 'saved' | ''>('')
  const [justCompletedObjectiveIds, setJustCompletedObjectiveIds] = useState<Set<number>>(new Set())

  const currentScreenBlob = useRef<Blob | null>(null)
  const messageIdCounter = useRef(0)
  const prevObjectivesRef = useRef<Objective[]>([])
  const nextObjectiveId = useRef(objectives.length > 0 ? Math.max(...objectives.map(o => o.id)) + 1 : 1)
  const gameScreenUrlRef = useRef<string | null>(null)
  
  const addChatMessage = useCallback((text: string, sender: 'user' | 'ai' | 'system') => {
    const message: ChatMessage = { id: messageIdCounter.current++, sender, text }
    setChatHistory(prev => [...prev, message])
  }, [])

  // Effect to detect and handle objective completion with race condition protection
  useEffect(() => {
    const prevObjectives = prevObjectivesRef.current

    // Prevent race conditions by checking if objectives have actually changed
    const objectivesChanged = JSON.stringify(prevObjectives) !== JSON.stringify(objectives)

    if (!objectivesChanged) return

    if (prevObjectives.length > 0 || objectives.some(o => o.completed)) {
        const newlyCompletedObjectives = objectives.filter(currentObj => {
            const prevObj = prevObjectives.find(p => p.id === currentObj.id)
            return currentObj.completed && (!prevObj || !prevObj.completed)
        })

        if (newlyCompletedObjectives.length > 0) {
            // Batch all objective completion notifications to prevent race conditions
            const completionTimeouts: NodeJS.Timeout[] = []

            newlyCompletedObjectives.forEach(obj => {
                addChatMessage(`Objective complete: '${obj.text}'`, 'system')
                setJustCompletedObjectiveIds(prev => new Set(prev).add(obj.id))

                const timeout = setTimeout(() => {
                    setJustCompletedObjectiveIds(prev => {
                        const newSet = new Set(prev)
                        newSet.delete(obj.id)
                        return newSet
                    })
                }, 2000)

                completionTimeouts.push(timeout)
            })

            // Cleanup function to clear timeouts if component unmounts or dependencies change
            return () => {
                completionTimeouts.forEach(timeout => clearTimeout(timeout))
            }
        }
    }

    prevObjectivesRef.current = objectives
  }, [objectives, addChatMessage])

  const handleReorderObjectives = (newObjectives: Objective[]) => {
    setObjectives(newObjectives)
  }

  const handleToggleObjective = (id: number) => {
    setObjectives(prev => prev.map(obj => obj.id === id ? { ...obj, completed: !obj.completed } : obj))
  }

  const handleDeleteObjective = (id: number) => {
    setObjectives(prev => prev.filter(obj => obj.id !== id))
  }

  const handleCreateObjective = (text: string) => {
    if (!text.trim()) return
    const newObjective: Objective = {
      id: nextObjectiveId.current++,
      text,
      completed: false
    }
    setObjectives(prev => [...prev, newObjective])
  }

  // Cleanup URL objects on unmount to prevent memory leaks
  useEffect(() => {
    return () => {
      if (gameScreenUrlRef.current) {
        URL.revokeObjectURL(gameScreenUrlRef.current)
        gameScreenUrlRef.current = null
      }
    }
  }, [])

  // Global error handler for unhandled promise rejections
  useEffect(() => {
    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      console.error('Unhandled promise rejection:', event.reason)
      addChatMessage('An unexpected error occurred. Please try again.', 'system')
      event.preventDefault()
    }

    window.addEventListener('unhandledrejection', handleUnhandledRejection)

    return () => {
      window.removeEventListener('unhandledrejection', handleUnhandledRejection)
    }
  }, [addChatMessage])

  // Load state from localStorage on initial mount
  useEffect(() => {
    try {
        const savedStateJSON = localStorage.getItem(SAVE_KEY)
        if (savedStateJSON) {
            const savedState = JSON.parse(savedStateJSON)
            
            if (savedState.appSettings) setAppSettings(savedState.appSettings)
            if (savedState.aiGoal) setAiGoal(savedState.aiGoal)
            if (savedState.objectives) {
              setObjectives(savedState.objectives)
              nextObjectiveId.current = savedState.objectives.length > 0 ? Math.max(...savedState.objectives.map((o: Objective) => o.id)) + 1 : 1
            }
            if (savedState.chatHistory) setChatHistory(savedState.chatHistory)
            if (savedState.actionHistory) setActionHistory(savedState.actionHistory)
            if (savedState.achievements) setAchievements(savedState.achievements)
            if (savedState.party) setCharacters(savedState.party || savedState.characters)
            if (savedState.inventory) setInventory(savedState.inventory)
            if (savedState.mapInfo) setMapInfo(savedState.mapInfo)
            if (savedState.playerStats) setPlayerStats(savedState.playerStats)
            
            addChatMessage('Session restored from previous state.', 'system')
        }
    } catch (error) {
        console.error("Failed to load state from localStorage", error)
        addChatMessage('Could not restore previous session.', 'system')
    }
  }, [addChatMessage])

  // Autosave state to localStorage
  const saveState = useCallback(() => {
    setSaveStatus('saving')
    try {
        const stateToSave = {
            appSettings,
            aiGoal,
            objectives,
            chatHistory,
            actionHistory,
            achievements,
            characters,
            inventory,
            mapInfo,
            playerStats
        }
        localStorage.setItem(SAVE_KEY, JSON.stringify(stateToSave))
        setSaveStatus('saved')
        setTimeout(() => setSaveStatus(''), 2000)
    } catch (error) {
        console.error("Failed to save state to localStorage", error)
        setSaveStatus('')
    }
  }, [appSettings, aiGoal, objectives, chatHistory, actionHistory, achievements, characters, inventory, mapInfo, playerStats])

  useInterval(saveState, 15000)

  useEffect(() => {
    const loadModels = async () => {
      if (
        (appSettings.aiProvider === 'google' && !appSettings.googleApiKey) ||
        (appSettings.aiProvider === 'openrouter' && !appSettings.openrouterApiKey) ||
        (appSettings.aiProvider === 'lmstudio' && !appSettings.lmStudioUrl)
      ) {
        setAvailableModels([])
        return
      }

      setIsModelListLoading(true)
      setAvailableModels([])
      try {
        const models = await fetchAvailableModels(appSettings)
        setAvailableModels(models)
        if (models.length > 0 && !models.find(m => m.id === appSettings.selectedModel)) {
            setAppSettings(prev => ({...prev, selectedModel: models[0].id }))
        }
      } catch (error) {
        console.error("Failed to load models:", error)
        addChatMessage(`Failed to load models for ${appSettings.aiProvider}.`, 'system')
      } finally {
        setIsModelListLoading(false)
      }
    }

    loadModels()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [appSettings.aiProvider, appSettings.googleApiKey, appSettings.openrouterApiKey, appSettings.lmStudioUrl])

  const refreshFullGameState = useCallback(async () => {
    if (!isRomLoaded) return
    try {
      const [screenBlob, characterData, gameState] = await Promise.all([
        getScreen(appSettings.backendUrl),
        getCharacters(appSettings.backendUrl),
        getGameState(appSettings.backendUrl)
      ])

      currentScreenBlob.current = screenBlob
      const newUrl = URL.createObjectURL(screenBlob)

      setGameScreenUrl(prevUrl => {
        if (prevUrl) {
          URL.revokeObjectURL(prevUrl)
        }
        gameScreenUrlRef.current = newUrl
        return newUrl
      })

      if (characterData) setCharacters(characterData)
      if (gameState) {
        setCharacters(gameState.characters)
        setAchievements(gameState.achievements)
        setInventory(gameState.inventory)
        setMapInfo(gameState.map)
        setPlayerStats(gameState.stats)
        setDialogue(gameState.dialogue || '')
      }

    } catch (error) {
      console.error(error)
      setAiState(AIState.IDLE)
      const errorMessage = error instanceof Error ? error.message : "An unknown error occurred."
      addChatMessage(`Connection Error: ${errorMessage}`, 'system')
    }
  }, [addChatMessage, isRomLoaded, appSettings.backendUrl])
  
  const refreshScreen = useCallback(async () => {
    if (!isRomLoaded) return
    try {
        const screenBlob = await getScreen(appSettings.backendUrl)
        currentScreenBlob.current = screenBlob
        const newUrl = URL.createObjectURL(screenBlob)

        setGameScreenUrl(prevUrl => {
            if (prevUrl) {
                URL.revokeObjectURL(prevUrl)
            }
            gameScreenUrlRef.current = newUrl
            return newUrl
        })
    } catch (error) {
        console.warn("Screen stream failed a frame:", error)
    }
  }, [isRomLoaded, appSettings.backendUrl])

  const aiTick = useCallback(async () => {
    if (aiState !== AIState.RUNNING || !isRomLoaded) return

    setAiState(AIState.THINKING)

    try {
        await refreshFullGameState()

        if (!currentScreenBlob.current) {
            console.error("No screen data available for AI to process.")
            addChatMessage("Could not retrieve game screen.", 'system')
            setAiState(AIState.RUNNING)
            return
        }

        const currentGoal = aiGoal || "Explore the world."
        const { reasoning, action } = await getAIThoughtAndAction(appSettings, currentScreenBlob.current, currentGoal, objectives, actionHistory, mapInfo, dialogue)

        setCurrentReasoning(reasoning)
        setActionHistory(prev => [...prev, action])

        await sendAction(appSettings.backendUrl, action)

        setAiState(AIState.RUNNING)

    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : "An unknown error occurred."
        console.error('AI tick operation failed:', error)
        addChatMessage(`AI operation failed: ${errorMessage}`, 'system')
        setCurrentReasoning(`Error: ${errorMessage}`)
        setAiState(AIState.RUNNING)
    }
  }, [aiState, actionHistory, refreshFullGameState, aiGoal, objectives, addChatMessage, isRomLoaded, appSettings, mapInfo, dialogue])

  // Main AI action loop
  useInterval(aiTick, aiState === AIState.RUNNING && isRomLoaded ? appSettings.aiActionInterval : null)
  
  // Constant screen refresh for visual feedback
  useInterval(refreshScreen, isRomLoaded && aiState !== AIState.THINKING ? STREAM_INTERVAL_MS : null)

  // Slower state refresh for panels when AI is idle
  useInterval(refreshFullGameState, aiState === AIState.IDLE && isRomLoaded ? IDLE_STATE_REFRESH_MS : null)

  const handleToggleAI = () => {
    if (!isRomLoaded) return
    if (aiState === AIState.IDLE) {
      setActionHistory([])
      setCurrentReasoning("AI is starting...")
      addChatMessage('AI starting...', 'system')
      setAiState(AIState.RUNNING)
    } else {
      addChatMessage('AI stopped by user.', 'system')
      setCurrentReasoning("AI stopped by user.")
      setAiState(AIState.IDLE)
    }
  }

  const handleSendMessage = async () => {
    if (!chatInput.trim() || isChatting) return

    const userMessage = chatInput
    addChatMessage(userMessage, 'user')
    setChatInput('')
    setIsChatting(true)

    try {
        const aiResponseText = await getChatResponse(appSettings, userMessage)
        addChatMessage(aiResponseText, 'ai')
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : "An unknown error occurred."
        console.error('Failed to get AI chat response:', error)
        addChatMessage(`AI chat failed: ${errorMessage}`, 'system')
    } finally {
        setIsChatting(false)
    }
  }

  const handleSaveSettings = (newSettings: AppSettings) => {
    if (newSettings.aiProvider !== appSettings.aiProvider) {
      newSettings.selectedModel = ''
    }
    setAppSettings(newSettings)
    setIsSettingsOpen(false)
    addChatMessage(`Settings updated. AI Provider: ${newSettings.aiProvider}.`, 'system')
  }

  const handleLoadRom = async (file: File) => {
    try {
        await loadRom(appSettings.backendUrl, file)
        setIsRomLoaded(true)
        setCurrentReasoning("Successfully loaded ROM. Ready for instructions.")
        addChatMessage(`Successfully loaded ${file.name}.`, 'system')
        await refreshFullGameState()
    } catch (error) {
        const message = error instanceof Error ? error.message : "An unknown error occurred."
        addChatMessage(`Failed to load ROM: ${message}`, 'system')
        setIsRomLoaded(false)
        throw error
    }
  }
  
  const handleSaveState = async () => {
    if (!isRomLoaded) return
    try {
      await saveGameState(appSettings.backendUrl)
      addChatMessage('Game state saved.', 'system')
    } catch (error) {
      const message = error instanceof Error ? error.message : "An unknown error occurred."
      addChatMessage(`Failed to save state: ${message}`, 'system')
    }
  }

  const handleLoadState = async () => {
    if (!isRomLoaded) return
    try {
      await loadGameState(appSettings.backendUrl)
      addChatMessage('Game state loaded.', 'system')
      await refreshFullGameState()
    } catch (error) {
      const message = error instanceof Error ? error.message : "An unknown error occurred."
      addChatMessage(`Failed to load state: ${message}`, 'system')
    }
  }

  return (
    <ErrorBoundary
      onError={(error, errorInfo) => {
        console.error('GLM4.5 UI ErrorBoundary caught an error:', error, errorInfo)
        // Optionally log to error reporting service
      }}
    >
      <div className="h-screen w-full flex flex-col bg-neutral-950 font-sans overflow-hidden">
        <Header
          achievements={achievements}
          onOpenSettings={() => setIsSettingsOpen(true)}
          runtime={playerStats.runtime}
          steps={playerStats.steps}
          saveStatus={saveStatus}
        />
        <main className="flex-grow grid grid-cols-1 md:grid-cols-12 gap-4 p-4 min-h-0 overflow-y-auto">
          <div className="md:col-span-4 lg:col-span-3 min-h-0">
            <ErrorBoundary>
              <AIPanel
                aiState={aiState}
                aiGoal={aiGoal}
                currentReasoning={currentReasoning}
                actionHistory={actionHistory}
                chatHistory={chatHistory}
                chatInput={chatInput}
                isChatting={isChatting}
                isRomLoaded={isRomLoaded}
                onScreenText={dialogue}
                onGoalChange={setAiGoal}
                onStart={handleToggleAI}
                onStop={handleToggleAI}
                onChatInputChange={setChatInput}
                onSendMessage={handleSendMessage}
              />
            </ErrorBoundary>
          </div>
          <div className="md:col-span-8 lg:col-span-5 min-h-0">
            <ErrorBoundary>
              <GamePanel
                characters={characters}
                gameScreenUrl={gameScreenUrl}
                lastAction={actionHistory.length > 0 ? actionHistory[actionHistory.length - 1] : null}
                isRomLoaded={isRomLoaded}
                onLoadRom={handleLoadRom}
                onSaveState={handleSaveState}
                onLoadState={handleLoadState}
              />
            </ErrorBoundary>
          </div>
          <div className="md:col-span-12 lg:col-span-4 min-h-0">
            <ErrorBoundary>
              <StatePanel
                objectives={objectives}
                inventory={inventory}
                inventoryTitle="INVENTORY (Key Items)"
                mapInfo={mapInfo}
                score={playerStats.score}
                onReorderObjectives={handleReorderObjectives}
                achievements={achievements}
                playerStats={playerStats}
                justCompletedObjectiveIds={justCompletedObjectiveIds}
                onToggleObjective={handleToggleObjective}
                onDeleteObjective={handleDeleteObjective}
                onCreateObjective={handleCreateObjective}
              />
            </ErrorBoundary>
          </div>
        </main>
        <ErrorBoundary>
          <SettingsModal
            isOpen={isSettingsOpen}
            onClose={() => setIsSettingsOpen(false)}
            currentSettings={appSettings}
            onSave={handleSaveSettings}
            availableModels={availableModels}
            isModelListLoading={isModelListLoading}
          />
        </ErrorBoundary>
      </div>
    </ErrorBoundary>
  )
}