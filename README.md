# OpenAI chat class for typescript

The `OpenAIWrapperClass`` provides a convenient and flexible way for developers to interact with OpenAI's API using the builder pattern. This class abstracts the complexity of handling various OpenAI API functionalities, such as text generation, vision, embeddings, speech, moderation, and image prompts.

One of the key benefits is the ease of configuration through fluent method chaining. Developers can set the OpenAI model, enable debug mode, specify parameters like temperature and timeout, and define custom tools for the chat. The class supports multiple OpenAI models, including the "gpt-3.5-turbo" and "gpt-4-1106-preview."

Using the class involves creating an instance with an API key and, optionally, a secret. Developers can then chain method calls to configure the OpenAI model and other parameters. The runPrompt and runPromptStream methods initiate the generation process, providing flexibility for both synchronous and streaming scenarios.

Here's an example of how to use the class to generate a chat completion:

```
const apiKey = "YOUR_OPENAI_API_KEY";
const openaiWrapper = new OpenAIWrapperClass(apiKey);

openaiWrapper
  .setModel("gpt-3.5-turbo")
  .setDebug(true)
  .setTemperature(0.8)
  .appendUserMessage("Tell me a joke.")
  .appendMessage({ role: "assistant", content: "Why did the chicken cross the road?" })
  .runPrompt({ seed: 100 })
  .then(() => {
    const lastResponse = openaiWrapper.getLastResponseAsMessageResult();
    console.log("Assistant's response:", lastResponse.content);
  })
  .catch((error) => {
    console.error("Error:", error.message);
  });
```

This example sets the model, enables debug mode, adds user and assistant messages to the chat, runs the prompt, and then retrieves and prints the assistant's response. The builder pattern facilitates a clean and expressive way to interact with OpenAI's API, making it suitable for a variety of natural language processing tasks.

The setters for calling the API are present as builders and also as options to provide to the `runPrompt` (and other) methods.

Besides creating the class, there is also integration with RXJS to allow for streamed responses. Maybe in future turn this into signals?

## Design objectives

Use Builder Pattern: The class employs the builder pattern, allowing developers to configure and customize interactions with OpenAI's API through method chaining.

Stay close to OpenAI: stay as close as possible to OpenAI's typings, and the JS SDK - for easy of maintenance

Configurability: Developers can easily set various options, including temperature, timeout, and maximum tokens, tailoring the behavior of the OpenAI API to specific needs.

Debugging Support: Includes a debug mode that, when enabled, provides additional information about the chat, tool calls, and other relevant details for debugging purposes.

Stream Interface using RXJS: Offers an interface for monitoring messages, stream delta, and concatenated stream, providing a reactive approach to handling responses.

Extensibility: Designed to be easily extensible, allowing for the addition of new features or adaptations to future changes in the OpenAI API.

## Beta APIs - Assistants

I have not implemented the Assistants API as it is in Beta. And it might require a total new class, as the APIs to Assistants are a group on their own.

## App.ts Examples:

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

# Documentation

See link [Documentation](./docs/index.html)
