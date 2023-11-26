import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import readline from "readline";
import { OpenAIWrapperClass, makeChatToolFunction } from "./openAIwrapper.class";

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

/*
Main example file
*/
(async () => {
  const newOpenAIthread = new OpenAIWrapperClass(OPENAI_API_KEY);

  // Example of how to use Dall-e
  if ((await askQuestion("Do you want to run Dall-e example (y/n) ")) === "y") {
    await newOpenAIthread
      .runImagePrompt("A painting of a person playing tennis", {
        model: "dall-e-2",
        response_format: "url",
        style: "natural",
      })
      .then((res) => res.getLastResponseAsImageResult())
      .then((res) => console.log(res));
  }

  // Example of how to base chat API
  if ((await askQuestion("Do you want to run prompt chat dialog example (y/n) ")) === "y") {
    await newOpenAIthread
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
      .then(async (res) => {
        console.log("Answer 1", await res.getLastResponseAsChatCompletionResult());
        return res;
      })
      .then((res) => res.appendUserMessage("Thanks. How are you?"))
      .then((ai) => ai.runPrompt())
      .then(async (res) => {
        console.log("Answer 2", await res.getLastResponseAsChatCompletionResult());
        console.log("All messages", newOpenAIthread.getMessages());
        return res;
      });
  }

  // streaming chat example
  if ((await askQuestion("Do you want to run stream chat example (y/n) ")) === "y") {
    if ((await askQuestion("Do you want to abort the stream after 1.5secs (y/n) ")) === "y")
      setTimeout(() => {
        console.log("Aborting stream");
        newOpenAIthread.abortStream();
      }, 1500);

    const subscription = newOpenAIthread.getStreamConcatedAsObservable().subscribe((res) => {
      console.log("Stream response", res);
    });

    await newOpenAIthread
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
      .runPromptStream()
      .then((res) => {
        subscription.unsubscribe();
        console.log("Last response", res.getLastResponseAsChatCompletionResult());
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

    await newOpenAIthread
      .setModel("gpt-3.5-turbo-1106")
      .setDebug(true)
      .setMessages([{ role: "user", content: "What's the weather like in San Francisco" }])
      .addToolWithFunction(getCurrentWeatherChatTool, getCurrentWeather)
      .runPrompt({})
      .then((res) => {
        console.log("Intermediate response", res.getLastResponseAsChatCompletionResult());
        console.log("All messages", res.getMessages());
        // we need to do a second run because tools need to run
        return res.runPrompt();
      })
      .then((res) => {
        console.log("Last response", res.getLastResponseAsChatCompletionResult());
        console.log("All messages", res.getMessages());
        return res;
      });
  }

  // vision - identification of objects in an image
  if ((await askQuestion("Do you want to run vision example (y/n) ")) === "y") {
    await newOpenAIthread
      .setMaxTokens(300)
      .setMessages([])
      .runVisionPrompt(
        "What do I see on this picture?",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
      )
      .then((res) => {
        console.log("Last response", res.getLastResponseAsVisionResult());
        console.log("Message", res.getMessages());
        return res;
      });
  }

  // creating embeddings
  if ((await askQuestion("Do you want to run embedding example (y/n) ")) === "y") {
    await newOpenAIthread.runEmbeddingPrompt("What is the meaning of life?").then((res) => {
      console.log("Last response", res.getLastResponseAsEmbbedingResult());
      return res;
    });
  }

  // generating speech
  if ((await askQuestion("Do you want to run speech example (y/n) ")) === "y") {
    const speechFile = path.resolve("./speech.mp3");
    const line = await askQuestion("Enter a sentence to convert to speech ");
    let res = await newOpenAIthread.runSpeechPrompt(line as string);
    await fs.promises.writeFile(speechFile, await res.getLastResponseAsSpeechBufferResult());

    console.log("File written to ./speech.mp3");
  }

  // moderation example
  if ((await askQuestion("Do you want to run moderation example (y/n) ")) === "y") {
    const line = await askQuestion("Enter a sentence to moderate: ");
    let res = await newOpenAIthread.runModerationPrompt(line as string);
    console.log("Moderation result", res.getLastResponseAsModerationResult());
  }

  rl.close();
  return;
})();
