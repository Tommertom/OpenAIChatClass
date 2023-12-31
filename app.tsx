import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import readline from "readline";
import { OpenAIWrapperClass, makeChatToolFunction } from "./openAIwrapper.class";

//

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function askQuestion(query: string) {
  return new Promise((resolve) => rl.question(query, (ans) => resolve(ans)));
}

dotenv.config();

const OPENAI_API_KEY = process.env.OPENAI_API_KEY as string;

// Example dummy function hard coded to return the same weather
// In production, this could be your backend API or an external API
function getCurrentWeather(location: string, unit: string = "fahrenheit"): string {
  console.log("called getCurrentWeather", location, unit);

  const weatherData: {
    [key: string]: { location: string; temperature: string; unit: string };
  } = {
    tokyo: { location: "Tokyo", temperature: "10", unit: "celsius" },
    "san francisco": {
      location: "San Francisco",
      temperature: "72",
      unit: "fahrenheit",
    },
    paris: { location: "Paris", temperature: "22", unit: "fahrenheit" },
  };

  const normalizedLocation = location.toLowerCase();
  const data = weatherData[normalizedLocation]
    ? weatherData[normalizedLocation]
    : { location, temperature: "unknown", unit: unit };

  return JSON.stringify(data);
}

/*
Main example file
*/
(async () => {
  const openAIthread = new OpenAIWrapperClass(OPENAI_API_KEY);

  // Example of how to use Dall-e
  if ((await askQuestion("Do you want to run Dall-e example (y/n) ")) === "y") {
    const line = await askQuestion("Enter the image prompt - ");
    await openAIthread
      .runImagePrompt(line as string, {
        model: "dall-e-3",
        response_format: "url",
        style: "vivid",
      })
      .then((ai) => ai.getLastResponseAsImageResult())
      .then((response) => console.log(response));
  }

  // Example of how to base chat API
  if ((await askQuestion("Do you want to run prompt chat dialog example (y/n) ")) === "y") {
    await openAIthread
      .setMessages([
        {
          role: "system",
          content:
            "You are a translator into German. The user will talk to you in English and you will answer in German",
        },
        {
          role: "user",
          content: "What do you think is the meaning of life?",
        },
      ])
      .runPrompt()
      .then(async (ai) => {
        console.log("Answer 1", await ai.getLastResponseAsChatCompletionResult());
        return ai;
      })
      .then((ai) => ai.appendUserMessage("Thanks. How are you?"))
      .then((ai) => ai.runPrompt())
      .then(async (res) => {
        console.log("Answer 2", await res.getLastResponseAsChatCompletionResult());
        console.log("All messages", openAIthread.getMessages());
        return res;
      });
  }

  // streaming chat example
  if ((await askQuestion("Do you want to run stream chat example (y/n) ")) === "y") {
    if ((await askQuestion("Do you want to abort the stream after 1.5secs (y/n) ")) === "y")
      setTimeout(() => {
        console.log("Aborting stream");
        openAIthread.abortStream();
      }, 1500);

    await openAIthread
      .setModel("gpt-3.5-turbo-1106")
      .setDebug(false)
      .setMaxTokens(30)
      .setMessages([
        {
          role: "system",
          content:
            "You are a translator into German. The user will give you a text in English  and you will provide the translation in German",
        },
        {
          role: "user",
          content: "Give a 100 word poem in German.",
        },
      ])
      .setStreamCallback((delta) => {
        console.log("Delta received", delta);
      })
      .runPromptStream()
      .then((ai) => {
        console.log("Last response", ai.getLastResponseAsChatCompletionResult());
      });
  }

  // using tools
  if ((await askQuestion("Do you want to run tool chat example (y/n) ")) === "y") {
    const getCurrentWeatherChatTool = makeChatToolFunction(
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

    await openAIthread
      .setModel("gpt-3.5-turbo-1106")
      .setDebug(true)
      .setMessages([{ role: "user", content: "What's the weather like in San Francisco" }])
      .addToolWithFunction(getCurrentWeatherChatTool, getCurrentWeather)
      .runPrompt({})
      .then((ai) => {
        console.log("Intermediate response", ai.getLastResponseAsChatCompletionResult());
        console.log("All messages", ai.getMessages());

        // we need to do a second run because tools need to run
        if (ai.needsToolRun()) return ai.runPrompt();
        else return ai;
      })
      .then((ai) => {
        console.log("Last response", ai.getLastResponseAsChatCompletionResult());
        console.log("All messages", ai.getMessages());

        if (ai.needsToolRun()) console.log("Tool needs to run again, but we are not doing it here");
        return ai;
      });
  }

  // vision - identification of objects in an image
  if ((await askQuestion("Do you want to run vision example (y/n) ")) === "y") {
    await openAIthread
      .setMaxTokens(300)
      .setMessages([])
      .runVisionPrompt(
        "Please solve the math problem on the picture, explain it step by step and give the answer.",
        "https://media.cheggcdn.com/media/0db/0db03581-c196-4706-b306-176f84c9d835/phpBp2Yfa.png"
      )
      .then((ai) => {
        console.log("Last response", ai.getLastResponseAsVisionResult());
        console.log("Message", ai.getMessages());
        return ai;
      });
  }

  // creating embeddings
  if ((await askQuestion("Do you want to run embedding example (y/n) ")) === "y") {
    await openAIthread.runEmbeddingPrompt("What is the meaning of life?").then((ai) => {
      console.log("Last response", ai.getLastResponseAsEmbbedingResult());
      return ai;
    });
  }

  // generating speech
  if ((await askQuestion("Do you want to run speech example (y/n) ")) === "y") {
    const speechFile = path.resolve("./speech.mp3");
    const line = await askQuestion("Enter a sentence to convert to speech ");
    let ai = await openAIthread.runSpeechPrompt(line as string);
    await fs.promises.writeFile(speechFile, await ai.getLastResponseAsSpeechBufferResult());

    console.log("File written to ./speech.mp3");
  }

  // moderation example
  if ((await askQuestion("Do you want to run moderation example (y/n) ")) === "y") {
    const line = await askQuestion("Enter a sentence to moderate: ");
    let ai = await openAIthread.runModerationPrompt(line as string);
    console.log("Moderation result", ai.getLastResponseAsModerationResult());
  }

  rl.close();
  return;
})();
