import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import { OpenAIChatThread, makeChatTool } from "./openai_chat";

dotenv.config();

const OPENAI_API_KEY = process.env.OPENAI_API_KEY as string;

// Example dummy function hard coded to return the same weather
// In production, this could be your backend API or an external API
function getCurrentWeather(location: string, unit = "fahrenheit") {
  console.log("called getCurrentWeather", location, unit);

  if (location.toLowerCase().includes("tokyo")) {
    return JSON.stringify({ location: "Tokyo", temperature: "10", unit: "celsius" });
  } else if (location.toLowerCase().includes("san francisco")) {
    return JSON.stringify({ location: "San Francisco", temperature: "72", unit: "fahrenheit" });
  } else if (location.toLowerCase().includes("paris")) {
    return JSON.stringify({ location: "Paris", temperature: "22", unit: "fahrenheit" });
  } else {
    return JSON.stringify({ location, temperature: "unknown" });
  }
}

(async () => {
  const newOpenAIthread = new OpenAIChatThread(OPENAI_API_KEY); // .runPrompt({ frequency_penalty: 0.5, presence_penalty: 0.5 });
  setTimeout(() => {
    // newOpenAIthread.abortStream();
  }, 1500);

  let res = await newOpenAIthread
    .setModel("gpt-3.5-turbo-1106")
    .setDebug(true)
    .setMaxTokens(30)
    .setMessages([
      {
        role: "system",
        content:
          "You are a translator into Albanian. The user will give you a text in English  and you will provide the translation in Albanian",
      },
      {
        role: "user",
        content: "Give a 100 word poem in Albanian.",
      },
    ])
    .runPromptStream();

  console.log(JSON.stringify(res, null, 2), JSON.stringify(newOpenAIthread.getMessages(), null, 2));

  let resa = await newOpenAIthread
    .appendMessage({
      role: "user",
      content: "I am fine, thank you. How are you?",
    })
    .runPromptWithMessageResponse();

  console.log(res);

  console.log("All messages", newOpenAIthread.getMessages());

  const chatTool = makeChatTool(
    "get_current_weather",
    "Get the current weather in a given location",
    {
      type: "object",
      properties: {
        location: {
          type: "string",
          description: "The city and state, e.g. San Francisco, CA",
        },
        unit: { type: "string", enum: ["celsius", "fahrenheit"] },
      },
      required: ["location"],
    }
  );

  let thread2 = await newOpenAIthread
    .setModel("gpt-3.5-turbo-1106")
    .setDebug(true)
    .setMessages([{ role: "user", content: "What's the weather like in San Francisco" }])
    .addToolWithFunction(chatTool, getCurrentWeather);

  let res2 = await thread2.runPrompt();
  console.log("addToolWithFunction res", JSON.stringify(res2, null, 2));

  console.log("Messages", JSON.stringify(thread2.getMessages(), null, 2));

  let res3 = await thread2.runPrompt();
  console.log("Second run with same thread", JSON.stringify(res3, null, 2));

  let res4 = await thread2
    .setMaxTokens(300)
    .runVisionPrompt(
      "What do I see on this picture?",
      "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    );

  console.log("runVisionVisionPrompt thread", JSON.stringify(res4, null, 2));

  console.log("Messages", JSON.stringify(thread2.getMessages(), null, 2));

  let res5 = await thread2.runEmbeddingPrompt("What is the meaning of life?");

  console.log("runembbedings", JSON.stringify(res5.length, null, 2));
  const speechFile = path.resolve("./speech.mp3");
  let res6 = await thread2.runSpeechPromptAsBuffer("What is the meaning of life?");

  await fs.promises.writeFile(speechFile, res6);

  let res7 = await thread2.runModerationPromptAsBoolean("I am an idiot");
  console.log("runModerationPrompt", JSON.stringify(res7, null, 2));
})();
