import { ChatCompletionMessageParam } from "openai/resources";
import { BehaviorSubject, Observable } from "rxjs";
import { CompletionCreateParamsBaseOptionals, OpenAIWrapperClass } from "./openAIwrapper.class";

export class OpenAIWrapperRXJSClass extends OpenAIWrapperClass {
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

  constructor(apiKey: string) {
    super(apiKey);
  }
  protected override _addmessages(messages: ChatCompletionMessageParam[]): OpenAIWrapperClass {
    super._addmessages(messages);
    this.messages = this.messages.concat(messages);
    this.messages$.next(this.messages);

    return this;
  }

  override async runPromptStream(
    deltaCallBackFn: (delta: string | undefined) => void,
    modelOptions?: CompletionCreateParamsBaseOptionals
  ): Promise<this> {
    let total = "";
    await super.runPromptStream((delta) => {
      this.streamDelta$.next(delta);
      total += delta;
      this.streamConcated$.next(total);

      deltaCallBackFn(delta);
    }, modelOptions);

    return this;
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
}
