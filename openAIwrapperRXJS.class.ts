import OpenAI from "openai";
import { hasOwn } from "openai/core";
import {
  ChatCompletionMessageParam,
  ChatCompletionTool,
  FunctionDefinition,
  FunctionParameters,
  ImageGenerateParams,
} from "openai/resources";
import { CompletionCreateParamsBase } from "openai/resources/completions";
import { BehaviorSubject, Observable } from "rxjs";

// can be more models than this
type SupportedOpenAIModels = "gpt-3.5-turbo" | "gpt-4-1106-preview" | "gpt-3.5-turbo-1106";

/**
 * Represents the count of tokens and tool calls in an OpenAI chat thread.
 */
export interface OpenAIWrapperClassCount {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  tool_calls: number;
}

/**
 * Represents optional parameters for creating a completion.
 */
export interface CompletionCreateParamsBaseOptionals
  extends Omit<CompletionCreateParamsBase, "prompt" | "model"> {}

/**
 * Optional parameters for generating an image.
 */
export interface ImageGenerateParamsOptionals extends Omit<ImageGenerateParams, "prompt"> {}

/**
 * Represents a chat thread for interacting with the OpenAI chat API.
 */
export class OpenAIWrapperClass {
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
  private threadCount: OpenAIWrapperClassCount = {
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
    tool_calls: 0,
  };

  // custom fields
  private receivedCompletions: Array<OpenAI.Chat.Completions.ChatCompletion> = [];
  private toolFunctionmap: Record<string, Function> = {};
  private debug: boolean = false;
  private lastResponse:
    | OpenAI.Images.ImagesResponse
    | Response
    | OpenAI.Moderations.Moderation
    | number[]
    | OpenAI.Chat.Completions.ChatCompletion
    | undefined = undefined;

  // the RXJS goodies
  private messages$: BehaviorSubject<ChatCompletionMessageParam[]> = new BehaviorSubject<
    ChatCompletionMessageParam[]
  >([]);

  private streamDelta$: BehaviorSubject<string | undefined> = new BehaviorSubject<
    string | undefined
  >(undefined);
  private streamConcated$: BehaviorSubject<string | undefined> = new BehaviorSubject<
    string | undefined
  >(undefined);

  /****************************************************************************************

    All configuration options for the OpenAIWrapperClass

  *****************************************************************************************/

  constructor(apiKey: string) {
    if (apiKey === undefined) throw new Error("API Key is required");
    this.openai = new OpenAI({ apiKey });
  }

  /**
   * Sets the model for the OpenAIWrapperClass.
   * @param model The supported OpenAI model.
   * @returns The updated OpenAIWrapperClass instance.
   */
  setModel(model: SupportedOpenAIModels): OpenAIWrapperClass {
    this.model = model;

    return this;
  }

  /**
   * Sets the debug mode for the OpenAIWrapperClass.
   * @param debug - A boolean value indicating whether debug mode should be enabled or disabled.
   * @returns The updated OpenAIWrapperClass instance.
   */
  setDebug(debug: boolean): OpenAIWrapperClass {
    this.debug = debug;

    return this;
  }

  /**
   * Sets the JSON mode for the OpenAIWrapperClass.
   * @param json_mode - A boolean value indicating whether to enable JSON mode.
   * @returns The updated OpenAIWrapperClass instance.
   */
  setJsonMode(json_mode: boolean): OpenAIWrapperClass {
    this.json_mode = json_mode;

    return this;
  }

  /**
   * Sets the temperature for generating responses in the OpenAI chat thread.
   * @param temperature The temperature value to set.
   * @returns The updated OpenAIWrapperClass instance.
   */
  setTemperature(temperature: number): OpenAIWrapperClass {
    this.temperature = temperature;

    return this;
  }

  /**
   * Sets the timeout for the OpenAIWrapperClass.
   * @param timeout The timeout value in milliseconds.
   * @returns The updated OpenAIWrapperClass instance.
   */
  setTimeout(timeout: number): OpenAIWrapperClass {
    this.timeout = timeout;

    return this;
  }

  /**
   * Sets the maximum number of tokens to generate in the response.
   *
   * @param max_tokens The maximum number of tokens.
   * @returns The updated OpenAIWrapperClass instance.
   */
  setMaxTokens(max_tokens: number): OpenAIWrapperClass {
    this.max_tokens = max_tokens;

    return this;
  }

  /**
   * Sets the functions for the OpenAIWrapperClass.
   *
   * @param functions - An array of ChatCompletionTool objects or FunctionDefinition objects.
   * @returns The updated OpenAIWrapperClass instance.
   */
  setTools(functions: ChatCompletionTool[] | FunctionDefinition[]): OpenAIWrapperClass {
    this.tools = [];
    functions.forEach((fn) => {
      this.addTool(fn);
    });

    return this;
  }

  /**
   * Appends a function to the list of functions in the OpenAIWrapperClass
   *
   * @see https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models
   *
   * @param chatToolFunction - The function to append.
   * @returns The updated OpenAIWrapperClass instance.
   */
  addTool(chatToolFunction: ChatCompletionTool | FunctionDefinition): OpenAIWrapperClass {
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
   * @see https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models
   *
   * @param chatToolFunction - The function definition as ChatCompletionTool or FunctionDefinition.
   * @param toolFunction - The tool function to associate with the function definition.
   * @returns The updated OpenAIWrapperClass instance.
   */
  addToolWithFunction(
    chatToolFunction: ChatCompletionTool | FunctionDefinition,
    toolFunctionInJS: Function
  ): OpenAIWrapperClass {
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
   * Sets the tool function map for the OpenAIWrapperClass.
   *
   * @param toolFunctionmap - The tool function map to set.
   * @returns The updated OpenAIWrapperClass instance.
   */
  setToolFunctionMap(toolFunctionmap: Record<string, Function>): OpenAIWrapperClass {
    this.toolFunctionmap = toolFunctionmap;

    return this;
  }

  /**
   * Sets a tool function for the OpenAIWrapperClass.
   *
   * @param name - The name of the tool function.
   * @param fn - The tool function to be set.
   * @returns The updated OpenAIWrapperClass instance.
   */
  setToolFunction(name: string, toolFunctionInJS: Function): OpenAIWrapperClass {
    this.toolFunctionmap[name] = toolFunctionInJS;

    return this;
  }

  /**
   * Retrieves the last response received.
   *
   * @returns The last response.
   */
  getLastResponse() {
    return this.lastResponse;
  }

  /****************************************************************************************

    TEXT GENERATION TEXT GENERATION TEXT GENERATION TEXT GENERATION TEXT GENERATION TEXT GENERATION
    
  *****************************************************************************************/

  /**
   * Sets the messages for the OpenAIWrapperClass.
   *
   * @param messages - An array of ChatCompletionMessageParam objects representing the messages to be set.
   * @returns The updated OpenAIWrapperClass instance.
   */
  setMessages(messages: ChatCompletionMessageParam[]): OpenAIWrapperClass {
    this.messages = [];
    this._addmessages(messages);

    return this;
  }

  /**
   * Appends a message to the chat conversation and run the prompt.
   *
   * @param message - The message to append.
   * @returns The updated OpenAIWrapperClass instance.
   */
  appendMessage(message: ChatCompletionMessageParam): OpenAIWrapperClass {
    this._addmessages([message]);

    return this;
  }

  /**
   * Appends a user message to the OpenAIWrapperClass.
   *
   * @param message - The message to be appended.
   * @returns The updated OpenAIWrapperClass instance.
   */
  appendUserMessage(message: string): OpenAIWrapperClass {
    this._addmessages([{ content: message, role: "user" }]);

    return this;
  }

  // private methods
  private _addmessages(messages: ChatCompletionMessageParam[]): OpenAIWrapperClass {
    this.messages = this.messages.concat(messages);
    this.messages$.next(messages);

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
        // reset the parameters for the stream
        let total = "";
        this.streamConcated$.next("");
        this.streamDelta$.next("");
        this.streamAbortController = response.controller;

        // let's emit the stream delta and the concated stream
        for await (const chunk of response) {
          const chunk_as_string = chunk.choices[0]?.delta?.content || "";
          total = total + chunk_as_string;

          this.streamDelta$.next(chunk_as_string);
          this.streamConcated$.next(total);

          if (this.debug) console.log("Stream output", chunk_as_string, total);
        }

        // if the stream is aborted, we need reset the last response and end
        if (this.streamAbortController === undefined) {
          this.lastResponse = undefined;

          return this;
        }

        // add this completion to the list of received completion once ready and not aborted
        if (this.streamAbortController !== undefined) {
          if (this.debug) console.log("Stream completed", total);

          this.lastResponse = {
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
          };

          this.receivedCompletions.push(this.lastResponse);

          this._addmessages([
            {
              role: "assistant",
              content: total,
            },
          ]);
        }

        // message that the stream is completed
        this.streamDelta$.next(undefined);
        this.streamConcated$.next(undefined);

        // make sure we cannot abort the stream anymore
        this.streamAbortController = undefined;

        return this;
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
   * Returns an Observable that emits the stream delta as a string.
   * @returns {Observable<string>} An Observable that emits the stream delta as a string.
   */
  getStreamDeltaAsObservable(): Observable<string | undefined> {
    return this.streamDelta$.asObservable();
  }

  /**
   * Returns an Observable that emits the concatenated stream as a string.
   * @returns An Observable that emits the concatenated stream as a string.
   */
  getStreamConcatedAsObservable(): Observable<string | undefined> {
    return this.streamConcated$.asObservable();
  }

  /**
   * Runs the prompt and returns a promise that resolves to the chat completion response.
   * @returns A promise that resolves to a `OpenAI.Chat.Completions.ChatCompletion` object.
   */
  async runPrompt(modelOptions?: CompletionCreateParamsBaseOptionals) {
    const response = await this.openai.chat.completions.create({
      ...modelOptions,
      messages: this.messages,
      model: this.model,
      stream: false,
      temperature: this.temperature,
      response_format: { type: this.json_mode ? "json_object" : "text" },
      tools: this.tools.length > 0 ? this.tools : undefined,
      max_tokens: this.max_tokens,
    });

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
            if (this.debug) console.log("Toolcall - calling function", functionName, functionArgs);
            const functionResponse = await functionToCall(functionArgs.location, functionArgs.unit);

            if (this.debug) console.log("Toolcall - function result", functionName, functionArgs);

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

    this.lastResponse = response;
    return this;
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
   * Returns the last response as a ChatCompletion object.
   * @returns {OpenAI.Chat.Completions.ChatCompletion} The last response as a ChatCompletion object.
   */
  getLastResponseAsChatCompletionResult() {
    return this.lastResponse as OpenAI.Chat.Completions.ChatCompletion;
  }

  /**
   * Retrieves the last result as a message result.
   * @returns The message result from the last chat completion result.
   */
  getLastResponseAsMessageResult() {
    return this.getLastResponseAsChatCompletionResult().choices[0].message;
  }

  /****************************************************************************************

    VISION VISION VISION VISION VISION VISION VISION VISION VISION VISION VISION VISION VISION 

  *****************************************************************************************/

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

        this.lastResponse = response;

        return this;
      });
  }

  getLastResponseAsVisionResult() {
    return this.lastResponse as OpenAI.Chat.Completions.ChatCompletion;
  }

  /****************************************************************************************

    EMBEDDINGS EMBEDDINGS EMBEDDINGS EMBEDDINGS EMBEDDINGS EMBEDDINGS EMBEDDINGS EMBEDDINGS 

  *****************************************************************************************/

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

      this.lastResponse = response.data[0].embedding;

      return this;
    });
  }

  /**
   * Returns the last response as an embedding result.
   * @returns {number[]} The last response as an array of numbers representing the embedding result.
   */
  getLastResponseAsEmbbedingResult() {
    return this.lastResponse as number[];
  }

  /****************************************************************************************

    SPEECH SPEECH SPEECH SPEECH SPEECH SPEECH SPEECH SPEECH SPEECH SPEECH SPEECH SPEECH SPEECH 

  *****************************************************************************************/

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

        this.lastResponse = response;

        return this;
      });
  }

  /**
   * Retrieves the last response as a speech result.
   *
   * @returns The last response as a speech result.
   */
  getLastResponseAsSpeechResult(): Response {
    return this.lastResponse as Response;
  }

  /**
   * Retrieves the last response as a speech buffer result.
   * @returns A promise that resolves to a Buffer containing the speech data.
   */
  getLastResponseAsSpeechBufferResult(): Promise<Buffer> {
    return (this.lastResponse as Response)
      .arrayBuffer()
      .then((arrayBuffer) => Buffer.from(arrayBuffer));
  }

  /****************************************************************************************

    MODERATION MODERATION MODERATION MODERATION MODERATION MODERATION MODERATION MODERATION

  *****************************************************************************************/

  /**
   * Runs a moderation prompt using the OpenAI API.
   * @param input The input string for the moderation prompt.
   * @returns A Promise that resolves to the moderation prompt response.
   */
  runModerationPrompt(input: string) {
    return this.openai.moderations.create({ input }).then((response) => {
      if (this.debug) console.log("runModerationPrompt response", response);
      this.lastResponse = response.results[0];
      return this;
    });
  }

  /**
   * Retrieves the last response as a Moderation result.
   * @returns The last response as a Moderation object.
   */
  getLastResponseAsModerationResult() {
    return this.lastResponse as OpenAI.Moderations.Moderation;
  }

  /**
   * Runs the moderation prompt and returns a boolean indicating whether the input was flagged.
   * @param input - The input string to be checked for moderation.
   * @returns A promise that resolves to a boolean indicating whether the input was flagged.
   */
  getLastResponseAsModerationBooleanResult() {
    const response = this.getLastResponseAsModerationResult();
    return response.flagged;
  }

  /****************************************************************************************

    IMAGE IMAGE IMAGE IMAGE IMAGE IMAGE IMAGE IMAGE IMAGE IMAGE IMAGE IMAGE IMAGE IMAGE IMAGE

  *****************************************************************************************/

  /**
   * Runs the image prompt generation using the OpenAI API.
   *
   * @param prompt - The prompt for generating the image.
   * @param modelOptions - Optional parameters for the image generation.
   * @returns A Promise that resolves to the instance of the OpenAIChat class.
   */
  runImagePrompt(prompt: string, modelOptions: ImageGenerateParamsOptionals) {
    return this.openai.images.generate({ ...modelOptions, prompt }).then((response) => {
      if (this.debug) console.log("runImagePrompt response", response);

      this.lastResponse = response;
      return this;
    });
  }

  /**
   * Retrieves the URL of the last response as an image result.
   * @returns The URL of the last response as an image result, or undefined if there is no last response or the URL is not available.
   */
  getLastResponseAsImageResult() {
    return (this.lastResponse as OpenAI.Images.ImagesResponse)?.data[0];
  }

  /****************************************************************************************

    private stuff private stuff private stuff private stuff private stuff private stuff

  *****************************************************************************************/

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
}

/**
 * Creates a ChatCompletionTool object.
 * @param name - The name of the tool.
 * @param description - The description of the tool.
 * @param paramaters - The parameters of the tool.
 * @returns A ChatCompletionTool object.
 */
export function makeChatToolFunction(
  name: string,
  description: string,
  parameters: FunctionParameters | StrictFuntionParameters
): ChatCompletionTool {
  return {
    type: "function",
    function: {
      name,
      description,
      parameters: parameters as OpenAI.FunctionParameters,
    },
  };
}

/**
 * Represents a stricter definition of function parameters for chat functions - as FunctionParameters is too loose
 */
export interface StrictFuntionParameters {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: {
      type: "object";
      properties: {
        [key: string]: {
          type: string | "string" | "integer";
          description: string;
          enum?: string[];
        };
      };
      required: string[];
    };
  };
}

/*

 {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    }
                },
                "required": ["location", "format", "num_days"]
            },
        }
    },



*/
