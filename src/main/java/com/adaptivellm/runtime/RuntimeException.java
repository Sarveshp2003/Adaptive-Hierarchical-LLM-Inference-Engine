package com.adaptivellm.runtime;


/**
 * Base exception for adaptive LLM runtime.
 *
 * All runtime-level failures should use this exception.
 */
public class RuntimeException extends Exception {


    private final ErrorCode errorCode;



    /**
     * Creates runtime exception.
     *
     * @param errorCode category of failure
     * @param message error description
     */
    public RuntimeException(
            ErrorCode errorCode,
            String message
    ) {

        super(message);

        this.errorCode = errorCode;
    }



    /**
     * Creates runtime exception with cause.
     *
     * @param errorCode category
     * @param message description
     * @param cause original exception
     */
    public RuntimeException(
            ErrorCode errorCode,
            String message,
            Throwable cause
    ) {

        super(message, cause);

        this.errorCode = errorCode;
    }



    /**
     * Returns error category.
     */
    public ErrorCode errorCode() {

        return errorCode;
    }



    @Override
    public String toString() {

        return "RuntimeException{" +
                "errorCode=" + errorCode +
                ", message=" + getMessage() +
                '}';
    }
}