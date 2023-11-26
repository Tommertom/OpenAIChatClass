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
  .runPrompt()
  .then(() => {
    const lastResponse = openaiWrapper.getLastResponseAsMessageResult();
    console.log("Assistant's response:", lastResponse.content);
  })
  .catch((error) => {
    console.error("Error:", error.message);
  });
```

This example sets the model, enables debug mode, adds user and assistant messages to the chat, runs the prompt, and then retrieves and prints the assistant's response. The builder pattern facilitates a clean and expressive way to interact with OpenAI's API, making it suitable for a variety of natural language processing tasks.

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

See link [docs](docs/index.html)
