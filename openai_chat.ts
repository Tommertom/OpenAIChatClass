import OpenAI from "openai";
import { hasOwn } from "openai/core";
import {
  ChatCompletionMessageParam,
  ChatCompletionTool,
  FunctionDefinition,
  FunctionParameters,
} from "openai/resources";
import { CompletionCreateParamsBase } from "openai/resources/completions";
import { BehaviorSubject, Observable } from "rxjs";

// https://github.com/openai/openai-node

type SupportedOpenAIModels = "gpt-3.5-turbo" | "gpt-4-1106-preview" | "gpt-3.5-turbo-1106";

export interface OpenAIChatThreadCount {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  tool_calls: number;
}

export interface CompletionCreateParamsBaseOptionals
  extends Omit<CompletionCreateParamsBase, "prompt" | "model"> {}

export class OpenAIChatThread {
  // the open AI interfaces
  private openai: OpenAI;
  private model: SupportedOpenAIModels = "gpt-3.5-turbo";

  private json_mode: boolean = false;
  private messages: ChatCompletionMessageParam[] = [];
  private tools: Array<ChatCompletionTool> = [];
  private temperature: number = 1;
  private timeout: number = 600000; // 10 minutes
  private max_tokens: number = 150;
  private streamAbortController: AbortController | undefined = undefined;

  // internal thread values
  private threadCount: OpenAIChatThreadCount = {
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
    tool_calls: 0,
  };

  // custom fields
  private receivedCompletions: Array<OpenAI.Chat.Completions.ChatCompletion> = [];
  private toolFunctionmap: Record<string, Function> = {};
  private debug: boolean = false;

  // the RXJS goodies
  private messages$: BehaviorSubject<ChatCompletionMessageParam[]> = new BehaviorSubject<
    ChatCompletionMessageParam[]
  >([]);

  private streamDelta$: BehaviorSubject<string> = new BehaviorSubject<string>("");
  private streamConcated$: BehaviorSubject<string> = new BehaviorSubject<string>("");

  constructor(apiKey: string, secret?: string) {
    if (apiKey === undefined) throw new Error("API Key is required");
    this.openai = new OpenAI({ apiKey });
  }

  /**
   * Sets the model for the OpenAIChatThread.
   * @param model The supported OpenAI model.
   * @returns The updated OpenAIChatThread instance.
   */
  setModel(model: SupportedOpenAIModels): OpenAIChatThread {
    this.model = model;
    return this;
  }

  /**
   * Sets the debug mode for the OpenAIChatThread.
   * @param debug - A boolean value indicating whether debug mode should be enabled or disabled.
   * @returns The updated OpenAIChatThread instance.
   */
  setDebug(debug: boolean): OpenAIChatThread {
    this.debug = debug;
    return this;
  }

  /**
   * Sets the JSON mode for the OpenAIChatThread.
   * @param json_mode - A boolean value indicating whether to enable JSON mode.
   * @returns The updated OpenAIChatThread instance.
   */
  setJsonMode(json_mode: boolean): OpenAIChatThread {
    this.json_mode = json_mode;
    return this;
  }

  /**
   * Sets the temperature for generating responses in the OpenAI chat thread.
   * @param temperature The temperature value to set.
   * @returns The updated OpenAIChatThread instance.
   */
  setTemperature(temperature: number): OpenAIChatThread {
    this.temperature = temperature;
    return this;
  }

  /**
   * Sets the timeout for the OpenAIChatThread.
   * @param timeout The timeout value in milliseconds.
   * @returns The updated OpenAIChatThread instance.
   */
  setTimeout(timeout: number): OpenAIChatThread {
    this.timeout = timeout;
    return this;
  }

  /**
   * Sets the maximum number of tokens to generate in the response.
   *
   * @param max_tokens The maximum number of tokens.
   * @returns The updated OpenAIChatThread instance.
   */
  setMaxTokens(max_tokens: number): OpenAIChatThread {
    this.max_tokens = max_tokens;
    return this;
  }

  /**
   * Sets the messages for the OpenAIChatThread.
   *
   * @param messages - An array of ChatCompletionMessageParam objects representing the messages to be set.
   * @returns The updated OpenAIChatThread instance.
   */
  setMessages(messages: ChatCompletionMessageParam[]): OpenAIChatThread {
    this.messages = [];
    this._addmessages(messages);
    return this;
  }

  // private methods
  private _addmessages(messages: ChatCompletionMessageParam[]): OpenAIChatThread {
    this.messages = this.messages.concat(messages);
    this.messages$.next(messages);
    return this;
  }

  /**
   * Appends a message to the chat conversation and run the prompt.
   *
   * @param message - The message to append.
   * @returns The updated OpenAIChatThread instance.
   */
  appendMessage(message: ChatCompletionMessageParam): OpenAIChatThread {
    this._addmessages([message]);
    return this;
  }

  /**
   * Appends a user message to the OpenAIChatThread.
   *
   * @param message - The message to be appended.
   * @returns The updated OpenAIChatThread instance.
   */
  appendUserMessage(message: string): OpenAIChatThread {
    this._addmessages([{ content: message, role: "user" }]);
    return this;
  }

  /**
   * Sets the functions for the OpenAIChatThread.
   *
   * @param functions - An array of ChatCompletionTool objects or FunctionDefinition objects.
   * @returns The updated OpenAIChatThread instance.
   */
  setTools(functions: ChatCompletionTool[] | FunctionDefinition[]): OpenAIChatThread {
    this.tools = [];
    functions.forEach((fn) => {
      this.addTool(fn);
    });
    return this;
  }

  /**
   * Appends a function to the list of functions in the OpenAIChatThread.
   *
   * @param fn - The function to append.
   * @returns The updated OpenAIChatThread instance.
   */
  addTool(chatToolFunction: ChatCompletionTool | FunctionDefinition): OpenAIChatThread {
    // if the function is a FunctionDefinition, we need to convert it to a ChatCompletionTool
    if (!hasOwn(chatToolFunction, "type")) {
      this.tools.push({
        type: "function",
        function: chatToolFunction as FunctionDefinition,
      });
    }
    if (hasOwn(chatToolFunction, "type")) {
      this.tools.push(chatToolFunction as ChatCompletionTool);
    }

    return this;
  }

  /**
   * Adds a tool with a function definition and associates it with a tool function.
   *
   * @param fn - The function definition as ChatCompletionTool or FunctionDefinition.
   * @param toolFunction - The tool function to associate with the function definition.
   * @returns The updated OpenAIChatThread instance.
   */
  addToolWithFunction(
    chatToolFunction: ChatCompletionTool | FunctionDefinition,
    toolFunctionInJS: Function
  ): OpenAIChatThread {
    this.addTool(chatToolFunction);

    // get the name of the function to associate with the tool function in the toolFunctionmap
    let name = "";
    if (hasOwn(chatToolFunction, "type")) {
      name = (chatToolFunction as ChatCompletionTool).function.name;
    } else {
      name = (chatToolFunction as FunctionDefinition).name;
    }
    this.setToolFunction(name, toolFunctionInJS);

    return this;
  }

  /**
   * Sets the tool function map for the OpenAIChatThread.
   *
   * @param toolFunctionmap - The tool function map to set.
   * @returns The updated OpenAIChatThread instance.
   */
  setToolFunctionMap(toolFunctionmap: Record<string, Function>): OpenAIChatThread {
    this.toolFunctionmap = toolFunctionmap;
    return this;
  }

  /**
   * Sets a tool function for the OpenAIChatThread.
   *
   * @param name - The name of the tool function.
   * @param fn - The tool function to be set.
   * @returns The updated OpenAIChatThread instance.
   */
  setToolFunction(name: string, toolFunctionInJS: Function): OpenAIChatThread {
    this.toolFunctionmap[name] = toolFunctionInJS;
    return this;
  }

  /**
   * Runs the prompt stream for generating completions.
   *
   * @param modelOptions - Optional parameters for the model.
   * @returns A promise that resolves to the response from the completion API.
   */
  runPromptStream(modelOptions?: CompletionCreateParamsBaseOptionals) {
    this.streamAbortController = undefined;
    return this.openai.chat.completions
      .create({
        ...modelOptions,
        messages: this.messages,
        model: this.model,
        stream: true,
        temperature: this.temperature,
        response_format: { type: this.json_mode ? "json_object" : "text" },
        tools: undefined,
      })
      .then(async (response) => {
        let total = "";
        this.streamConcated$.next(total);
        this.streamAbortController = response.controller;

        for await (const chunk of response) {
          const chunk_as_string = chunk.choices[0]?.delta?.content || "";
          this.streamDelta$.next(chunk_as_string);
          total = total + chunk_as_string;
          this.streamConcated$.next(total);
          if (this.debug) console.log("Stream output", chunk_as_string, total);
        }

        // add this completion to the list of received completion once ready and not aborted
        if (this.streamAbortController !== undefined) {
          if (this.debug) console.log("Stream completed", total);

          this.receivedCompletions.push({
            id: "stream",
            choices: [
              {
                finish_reason: "stop",
                index: 0,
                message: {
                  content: total,
                  role: "assistant",
                },
              },
            ],
            created: Date.now(),
            model: this.model,
            object: "chat.completion",

            usage: {
              completion_tokens: 0,
              prompt_tokens: 0,
              total_tokens: 0,
            },
          });

          this._addmessages([
            {
              role: "assistant",
              content: total,
            },
          ]);
        }

        this.streamAbortController = undefined;
        return response;
      });
  }

  /**
   * Aborts the stream if it is currently active.
   */
  abortStream() {
    if (this.streamAbortController !== undefined) {
      this.streamAbortController.abort();
      this.streamAbortController = undefined;

      if (this.debug) console.log("Stream aborted");
    }
  }

  /**
   * Runs the prompt and returns a promise that resolves to the chat completion response.
   * @returns A promise that resolves to a `OpenAI.Chat.Completions.ChatCompletion` object.
   */
  runPrompt(
    modelOptions?: CompletionCreateParamsBaseOptionals
  ): Promise<OpenAI.Chat.Completions.ChatCompletion> {
    return this.openai.chat.completions
      .create({
        ...modelOptions,
        messages: this.messages,
        model: this.model,
        stream: false,
        temperature: this.temperature,
        response_format: { type: this.json_mode ? "json_object" : "text" },
        tools: this.tools.length > 0 ? this.tools : undefined,
        max_tokens: this.max_tokens,
      })
      .then((response) => {
        // add this completion to the list of received completions
        this.receivedCompletions.push(response);
        if (this.debug) console.log("receivedCompletions length", this.receivedCompletions.length);

        // let's add the assistant response to the messages
        response.choices.forEach(async (choice) => {
          this._addmessages([choice.message]);

          // extend the messagethreath with the tool calls
          const responseMessage = choice.message;
          const toolCalls = responseMessage.tool_calls;
          if (toolCalls !== undefined) {
            if (this.debug) console.log("Tools to call - ", toolCalls);

            this.threadCount.tool_calls += toolCalls.length;

            // taken from https://platform.openai.com/docs/guides/function-calling
            for (const toolCall of toolCalls) {
              const functionName = toolCall.function.name;
              const functionToCall = this.toolFunctionmap[functionName];
              const functionArgs = JSON.parse(toolCall.function.arguments);

              if (functionToCall === undefined)
                throw new Error(`Function ${functionName} is not defined`);

              if (functionToCall !== undefined) {
                if (this.debug)
                  console.log("Toolcall - calling function", functionName, functionArgs);
                const functionResponse = await functionToCall(
                  functionArgs.location,
                  functionArgs.unit
                );

                if (this.debug)
                  console.log("Toolcall - function result", functionName, functionArgs);

                this.messages.push({
                  tool_call_id: toolCall.id,
                  role: "tool",
                  //@ts-ignore
                  name: functionName,
                  content: functionResponse,
                });
              }
            }
          }
        });

        // update the counters
        this._updateThreadCount(response);

        // debug output
        if (this.debug) this._showPromptDebugInfo();

        return response;
      });
  }

  /**
   * Runs the prompt and returns the message response.
   * @returns A promise that resolves to the message response.
   */
  runPromptWithMessageResponse(modelOptions?: CompletionCreateParamsBaseOptionals) {
    return this.runPrompt(modelOptions).then((response) => {
      return response.choices[0].message;
    });
  }

  /**
   * Runs a vision prompt using the OpenAI chat completions API.
   *
   * @param text - The text prompt provided by the user.
   * @param image_url - The URL of the image to be used in the prompt.
   * @param detail - The level of detail for the image processing. Can be "auto", "low", or "high". Optional.
   * @returns A Promise that resolves to the response from the OpenAI chat completions API.
   */
  runVisionPrompt(
    text: string,
    image_url: string,
    detail?: "auto" | "low" | "high",
    modelOptions?: CompletionCreateParamsBaseOptionals
  ) {
    return this.openai.chat.completions
      .create({
        ...modelOptions,
        model: "gpt-4-vision-preview",
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: text },
              { type: "image_url", image_url: { url: image_url, detail: detail } },
            ],
          },
        ],
        stream: false,
        temperature: this.temperature,
        max_tokens: this.max_tokens,
      })
      .then((response) => {
        // add this completion to the list of received completions
        this.receivedCompletions.push(response);
        if (this.debug) console.log("receivedCompletions length", this.receivedCompletions.length);

        // let's add the assistant response to the messages
        response.choices.forEach(async (choice) => {
          this._addmessages([choice.message]);
        });

        // update the counters
        this._updateThreadCount(response);

        // debug output
        if (this.debug) this._showPromptDebugInfo();

        return response;
      });
  }

  /**
   * Runs an embedding prompt using the OpenAI API.
   * @param input The input string for the prompt.
   * @param model The model to use for the embedding. Defaults to "text-embedding-ada-002".
   * @returns A Promise that resolves to the result of the embedding prompt.
   */
  runEmbeddingPrompt(input: string, model: string = "text-embedding-ada-002") {
    return this.openai.embeddings.create({ input, model }).then((response) => {
      this._updateThreadCount(response as unknown as OpenAI.Chat.Completions.ChatCompletion);
      this._showPromptDebugInfo();
      return response.data[0].embedding;
    });
  }

  /**
   * Runs a speech prompt using the OpenAI API.
   *
   * @see https://platform.openai.com/docs/guides/text-to-speech?lang=node
   * @see https://platform.openai.com/docs/api-reference/embeddings/create?lang=node.js
   *
   * @param input The input text for the speech prompt.
   * @param model The model to use for generating the speech. Can be a string or one of the predefined values: "tts-1", "tts-1-hd".
   * @param voice The voice to use for generating the speech. Can be one of the predefined values: "alloy", "echo", "fable", "onyx", "nova", "shimmer".
   * @param response_format (Optional) The format of the speech response. Can be one of the predefined values: "mp3", "opus", "aac", "flac".
   * @param speed (Optional) The speed of the generated speech. A number between 0.1 and 3.0.
   * @returns A Promise that resolves to the speech response.
   */
  runSpeechPrompt(
    input: string,
    model: (string & {}) | "tts-1" | "tts-1-hd" = "tts-1",
    voice: "alloy" | "echo" | "fable" | "onyx" | "nova" | "shimmer" = "alloy",
    response_format: "mp3" | "opus" | "aac" | "flac" = "mp3",
    speed?: number
  ) {
    return this.openai.audio.speech
      .create({
        input,
        model,
        voice,
        response_format,
        speed,
      })
      .then((response) => {
        if (this.debug) console.log("runSpeechPrompt response", response);
        return response;
      });
  }

  /**
   * Runs a speech prompt and returns the response as an ArrayBuffer.
   * @param input - The input prompt.
   * @param model - The model to use for generating the response.
   * @param voice - The voice to use for the speech.
   * @param response_format - The format of the response audio (optional).
   * @param speed - The speed of the response audio (optional).
   * @returns A Promise that resolves to an ArrayBuffer containing the response audio.
   */
  runSpeechPromptAsBuffer(
    input: string,
    model: (string & {}) | "tts-1" | "tts-1-hd" = "tts-1",
    voice: "alloy" | "echo" | "fable" | "onyx" | "nova" | "shimmer" = "alloy",
    response_format: "mp3" | "opus" | "aac" | "flac" = "mp3",
    speed?: number
  ) {
    return this.runSpeechPrompt(input, model, voice, response_format, speed).then((response) => {
      return response.arrayBuffer().then((arrayBuffer) => Buffer.from(arrayBuffer));
    });
  }

  /**
   * Runs a moderation prompt using the OpenAI API.
   * @param input The input string for the moderation prompt.
   * @returns A Promise that resolves to the moderation prompt response.
   */
  runModerationPrompt(input: string) {
    return this.openai.moderations.create({ input }).then((response) => {
      if (this.debug) console.log("runModerationPrompt response", response);
      return response.results[0];
    });
  }

  /**
   * Runs the moderation prompt and returns a boolean indicating whether the input was flagged.
   * @param input - The input string to be checked for moderation.
   * @returns A promise that resolves to a boolean indicating whether the input was flagged.
   */
  runModerationPromptAsBoolean(input: string) {
    return this.runModerationPrompt(input).then((response) => {
      return response.flagged;
    });
  }

  /**
   * Displays debug information about the prompt.
   */
  private _showPromptDebugInfo() {
    console.log("showPromptDebugInfo messages", JSON.stringify(this.messages, null, 2));
    console.log("showPromptDebugInfo threadCount", this.threadCount);
    console.log("showPromptDebugInfo receivedCompletions", this.receivedCompletions);
  }

  /**
   * Updates the thread count based on the response from OpenAI Chat API.
   * @param response The response object from OpenAI Chat API.
   */
  private _updateThreadCount(response: OpenAI.Chat.Completions.ChatCompletion) {
    if (response.usage !== undefined) {
      const { prompt_tokens, completion_tokens, total_tokens } = response.usage;
      if (prompt_tokens !== undefined) this.threadCount.prompt_tokens += prompt_tokens;
      if (completion_tokens !== undefined) this.threadCount.completion_tokens += completion_tokens;
      if (total_tokens !== undefined) this.threadCount.total_tokens += total_tokens;
    }
  }

  /**
   * Retrieves the messages stored in the chat.
   * @returns An array of ChatCompletionMessageParam objects representing the messages.
   */
  getMessages(): ChatCompletionMessageParam[] {
    return this.messages;
  }

  /**
   * Returns an Observable that emits an array of ChatCompletionMessageParam objects whenever there is a new message.
   * @returns {Observable<ChatCompletionMessageParam[]>} The Observable that emits the array of ChatCompletionMessageParam objects.
   */
  getMessagesAsObservable(): Observable<ChatCompletionMessageParam[]> {
    return this.messages$.asObservable();
  }

  /**
   * Returns an Observable that emits the stream delta as a string.
   * @returns {Observable<string>} An Observable that emits the stream delta as a string.
   */
  getStreamDeltaAsObservable(): Observable<string> {
    return this.streamDelta$.asObservable();
  }

  /**
   * Returns an Observable that emits the concatenated stream as a string.
   * @returns An Observable that emits the concatenated stream as a string.
   */
  getStreamConcatedAsObservable(): Observable<string> {
    return this.streamConcated$.asObservable();
  }
}

/**
 * Creates a ChatCompletionTool object.
 * @param name - The name of the tool.
 * @param description - The description of the tool.
 * @param paramaters - The parameters of the tool.
 * @returns A ChatCompletionTool object.
 */
export function makeChatTool(
  name: string,
  description: string,
  parameters: FunctionParameters
): ChatCompletionTool {
  return {
    type: "function",
    function: {
      name,
      description,
      parameters,
    },
  };
}
