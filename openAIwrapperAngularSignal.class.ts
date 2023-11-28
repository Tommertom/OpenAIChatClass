import { Signal, WritableSignal, signal } from "@angular/core";
import { ChatCompletionMessageParam } from "openai/resources";
import { CompletionCreateParamsBaseOptionals, OpenAIWrapperClass } from "./openAIwrapper.class";

/**
 * Represents a class that extends OpenAIWrapperClass and provides additional functionality for handling API out using RXJS
 *
 */
export class OpenAIWrapperSignalClass extends OpenAIWrapperClass {
  /**
   * Represents the messages$ BehaviorSubject.
   * It emits the current value and any subsequent changes to its subscribers.
   */
  private messages$: WritableSignal<ChatCompletionMessageParam[]> = signal<
    ChatCompletionMessageParam[]
  >([]);

  /**
   * Represents the streamDelta$ BehaviorSubject.
   * It emits the current value and any subsequent changes to its subscribers.
   */
  private streamDelta$: WritableSignal<string | undefined> = signal<string | undefined>(undefined);

  /**
   * Represents the streamConcated$ BehaviorSubject.
   * It emits the current value and any subsequent changes to its subscribers.
   */
  private streamConcated$: WritableSignal<string | undefined> = signal<string | undefined>(
    undefined
  );

  constructor(apiKey: string) {
    super(apiKey);
  }

  /**
   * Adds messages to the chat completion and updates the messages subject. Overridden so we capture the events of adding new stuff
   * @param messages The messages to be added.
   * @returns The updated instance of the class.
   */
  protected override _addmessages(messages: ChatCompletionMessageParam[]): this {
    super._addmessages(messages);
    this.messages$.set(this.messages);

    return this;
  }

  /**
   * Runs the prompt stream and sets up the necessary callbacks to capture the stream delta and the concatenated stream using RXJS
   *
   * @param modelOptions - Optional parameters for the model.
   * @returns A promise that resolves to the current instance of the class.
   */
  override async runPromptStream(
    modelOptions?: CompletionCreateParamsBaseOptionals
  ): Promise<this> {
    let total = "";

    this.setStreamCallback((delta) => {
      this.streamDelta$.set(delta);
      total += delta;
      this.streamConcated$.set(total);
    });

    await super.runPromptStream(modelOptions);

    return this;
  }

  /**
   * Returns an Observable that emits the stream delta as a string or undefined.
   * @returns An Observable that emits the stream delta as a string or undefined.
   */
  getStreamDeltaAsObservable(): Signal<string | undefined> {
    return this.streamDelta$.asReadonly();
  }

  /**
   * Returns an Observable that emits the concatenated stream as a string.
   * @returns An Observable that emits the concatenated stream as a string.
   */
  getStreamConcatedAsObservable(): Signal<string | undefined> {
    return this.streamConcated$.asReadonly();
  }
}
