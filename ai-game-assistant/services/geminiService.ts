import { GoogleGenAI } from "@google/genai";
import type { GameAction } from '../types';

// Get API key from localStorage or environment variable
const apiKey = localStorage.getItem('geminiApiKey') || process.env.API_KEY || '';

if (!apiKey) {
  console.warn("No API key found. AI functionality will not work. Please set your Gemini API key in Settings.");
}

let ai: GoogleGenAI | null = null;
if (apiKey) {
  ai = new GoogleGenAI({ apiKey });
}

const VALID_ACTIONS: ReadonlySet<GameAction> = new Set(['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT']);

/**
 * Converts an image URL to a base64 string.
 * This is a client-side implementation.
 */
export async function urlToBase64(url: string): Promise<string> {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const blob = await response.blob();
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                if (typeof reader.result === 'string') {
                    resolve(reader.result.split(',')[1]);
                } else {
                    reject(new Error('Failed to read blob as base64 string.'));
                }
            };
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    } catch (error) {
        console.error("Error converting URL to Base64:", error);
        throw error;
    }
}

/**
 * Gets the current emulator screen from the backend server
 */
export async function getEmulatorScreen(): Promise<string> {
    try {
        const response = await fetch('http://localhost:5000/api/screen');
        if (!response.ok) {
            if (response.status === 400) {
                throw new Error("No ROM loaded in the emulator. Please load a ROM first.");
            }
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        return data.image;
    } catch (error) {
        console.error("Error getting emulator screen:", error);
        throw error;
    }
}


export async function getAINextMove(
  imageBase64: string,
  goal: string,
  history: string[]
): Promise<GameAction> {
  if (!ai) {
    throw new Error("AI not initialized. Please set your Gemini API key in Settings.");
  }
  
  const prompt = `You are an expert AI playing a retro video game. Your current high-level objective is: "${goal}".
Review the last few actions taken: ${history.slice(-5).join(', ')}.
Based on the provided game screenshot, determine the single next button press to advance toward the objective.
Your response MUST be one of the following exact words: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT.
Do not provide any explanation or other text.`;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: 'image/jpeg',
              data: imageBase64,
            },
          },
          { text: prompt },
        ],
      },
    });

    const action = response.text.trim().toUpperCase() as GameAction;

    if (VALID_ACTIONS.has(action)) {
      return action;
    } else {
      console.warn(`Gemini returned an invalid action: "${action}". Defaulting to SELECT.`);
      return 'SELECT'; // Return a default safe action
    }
  } catch (error) {
    console.error("Error calling Gemini API:", error);
    throw new Error("Failed to get next move from AI.");
  }
}


export async function getAIChatResponse(
  imageBase64: string,
  goal: string,
  actionHistory: string[],
  userMessage: string,
  chatHistory: { sender: 'user' | 'ai', text: string }[]
): Promise<string> {
  if (!ai) {
    throw new Error("AI not initialized. Please set your Gemini API key in Settings.");
  }
  
  const historyText = chatHistory.map(m => `${m.sender === 'user' ? 'User' : 'AI'}: ${m.text}`).join('\n');

  const prompt = `You are an AI game assistant observing a game.
The user's high-level objective is: "${goal}".
The most recent actions taken were: [${actionHistory.slice(-5).join(', ')}].

Here is the recent chat history:
${historyText}

Now, the user says: "${userMessage}"

Based on the current game screen, the objective, and the history, provide a helpful and concise response to the user's message.`;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: 'image/jpeg',
              data: imageBase64,
            },
          },
          { text: prompt },
        ],
      },
    });

    return response.text.trim();
  } catch (error) {
    console.error("Error calling Gemini API for chat:", error);
    throw new Error("Failed to get chat response from AI.");
  }
}