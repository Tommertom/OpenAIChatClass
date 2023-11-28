import { ChatCompletionMessageParam } from "openai/resources";
import { BehaviorSubject, Observable } from "rxjs";
import { CompletionCreateParamsBaseOptionals, OpenAIWrapperClass } from "./openAIwrapper.class";

/**
 * Represents a class that extends OpenAIWrapperClass and provides additional functionality for handling API out using RXJS
 *
 */
export class OpenAIWrapperRXJSClass extends OpenAIWrapperClass {
  /**
   * Represents the messages$ BehaviorSubject.
   * It emits the current value and any subsequent changes to its subscribers.
   */
  private messages$: BehaviorSubject<ChatCompletionMessageParam[]> = new BehaviorSubject<
    ChatCompletionMessageParam[]
  >([]);

  /**
   * Represents the streamDelta$ BehaviorSubject.
   * It emits the current value and any subsequent changes to its subscribers.
   */
  private streamDelta$: BehaviorSubject<string | undefined> = new BehaviorSubject<
    string | undefined
  >(undefined);

  /**
   * Represents the streamConcated$ BehaviorSubject.
   * It emits the current value and any subsequent changes to its subscribers.
   */
  private streamConcated$: BehaviorSubject<string | undefined> = new BehaviorSubject<
    string | undefined
  >(undefined);

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
    this.messages$.next(this.messages);

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
      this.streamDelta$.next(delta);
      total += delta;
      this.streamConcated$.next(total);
    });

    await super.runPromptStream(modelOptions);

    return this;
  }

  /**
   * Returns an Observable that emits the stream delta as a string or undefined.
   * @returns An Observable that emits the stream delta as a string or undefined.
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
}
