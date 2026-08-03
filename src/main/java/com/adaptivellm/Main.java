package com.adaptivellm;

import com.adaptivellm.runtime.RuntimeBootstrap;
import com.adaptivellm.runtime.RuntimeEngine;


/**
 * Adaptive LLM Runtime entry point.
 */
public final class Main {


    public static void main(
            String[] args
    ) {


        RuntimeEngine engine = null;


        try {


            engine =
                    RuntimeBootstrap.start();



            System.out.println(
                    "Adaptive LLM Runtime started"
            );



            /*
             * Future:
             *
             * Start inference server
             *
             * Accept requests
             *
             */



        }
        catch(Exception e) {


            e.printStackTrace();


        }
        finally {


            RuntimeBootstrap.shutdown(
                    engine
            );
        }
    }
}