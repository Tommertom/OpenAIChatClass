# OpenAI chat class for typescript

The `OpenAIWrapperClass` typescript class provides a convenient and flexible way for developers to interact with OpenAI's API using the builder pattern. This class abstracts the complexity of handling various OpenAI API functionalities, such as text generation, vision, embeddings, speech, moderation, and image prompts.

One of the key benefits is the ease of configuration through fluent method chaining. Developers can set the OpenAI model, enable debug mode, specify parameters like temperature and timeout, and define custom tools for the chat. The class supports multiple OpenAI models, including the "gpt-3.5-turbo" and "gpt-4-1106-preview."

Using the class involves creating an instance with an API key. Developers can then chain method calls to configure the OpenAI model and other parameters. The runPrompt and runPromptStream methods initiate the generation process, providing flexibility for both synchronous and streaming scenarios.

Here's an example of how to use the class to generate a chat completion:

```
const apiKey = "YOUR_OPENAI_API_KEY";
const openAIthread = new OpenAIWrapperClass(OPENAI_API_KEY);

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
      console.log("Answer 1", await res.getLastResponseAsChatCompletionResult());
      return ai;
    })
    .then((ai) => ai.appendUserMessage("Thanks. How are you?"))
    .then((ai) => ai.runPrompt())
    .then(async (ai) => {
      console.log("Answer 2", await ai.getLastResponseAsChatCompletionResult());
      console.log("All messages", ai.getMessages());
      return ai;
    });
```

This example sets the model, enables debug mode, adds user and assistant messages to the chat, runs the prompt, and then retrieves and prints the assistant's response. The builder pattern facilitates a clean and expressive way to interact with OpenAI's API, making it suitable for a variety of natural language processing tasks.

The setters for calling the API are present as builders and also as options to provide to the `runPrompt` (and other) methods.

Another example - with helper functions to generate the tool (taken from the OpenAI docs)

```

// the typescript function that in production should call the backend
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


// let's generate the JSON we need to include in the call to OpenAI
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

const apiKey = "YOUR_OPENAI_API_KEY";
const openAIthread = new OpenAIWrapperClass(OPENAI_API_KEY);

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

    if (ai.)
    return ai.runPrompt();
  })
  .then((ai) => {
    console.log("Last response", ai.getLastResponseAsChatCompletionResult());
    console.log("All messages", ai.getMessages());
    return ai;
  });
```

And a stream example, where we use a callback to receive the intermediate results. And have a method to abort the stream if need be

```

const apiKey = "YOUR_OPENAI_API_KEY";
const openAIthread = new OpenAIWrapperClass(OPENAI_API_KEY);


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
  .then((res) => {
    console.log("Last response", res.getLastResponseAsChatCompletionResult());
  });
```

The class `OpenAIWrapperRXJSClass` adds RXJS capabilities to the message streams arising from API calls.

## Design objectives

Use Builder Pattern: The class employs the builder pattern, allowing developers to configure and customize interactions with OpenAI's API through method chaining.

Stay close to OpenAI: stay as close as possible to OpenAI's typings, and the JS SDK - for easy of maintenance

Configurability: Developers can easily set various options, including temperature, timeout, and maximum tokens, tailoring the behavior of the OpenAI API to specific needs.

Debugging Support: Includes a debug mode that, when enabled, provides additional information about the chat, tool calls, and other relevant details for debugging purposes.

Stream Interface using callback function: Offers an interface for monitoring messages, stream delta, and concatenated stream, providing a reactive approach to handling responses.

Extensibility: Designed to be easily extensible, allowing for the addition of new features or adaptations to future changes in the OpenAI API.

## App.ts Examples:

See [App.tsx](./App.tsx) to see the various OpenAI endpoints in action using the builder pattern.

DALL-E Example:
Uses DALL-E to generate an image based on a prompt.

Prompt Chat Dialog Example:
Demonstrates a chat dialog with the OpenAI chat model, including system and user messages.

Stream Chat Example:
Illustrates streaming chat with the option to abort after a specified duration.

Tool Chat Example:
Integrates a custom tool (getCurrentWeatherChatTool) for querying current weather information based on user input.

Vision Example:
Identifies objects in an image using the vision model.

Embedding Example:
Generates embeddings for a given text prompt.

Speech Example:
Converts user-provided text to speech and saves it as an MP3 file.

Moderation Example:
Moderates content based on user input.

## Beta APIs - Assistants

I have not implemented the Assistants API as it is in Beta. And it might require a total new class, as the APIs to Assistants are a group on their own.

## Documentation

See link [Documentation](./docs)
