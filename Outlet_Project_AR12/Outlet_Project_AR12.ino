 #include <Bridge.h>
#include <BridgeServer.h>
#include <YunClient.h>
#include <BridgeClient.h>
#include <HttpClient.h>

BridgeClient client;
BridgeServer toggle;

char server[] = "192.168.0.10";

String serverStr;
String command;

String data;

int mVperAmp = 66;
double voltage;
double VRMS;
double ampsRMS;
double current;
 
unsigned long lastMillis = 0;
unsigned int currentMillis = 0;
long interval = 1000;

void setup() 
{ 
//  // Bridge startup
//  pinMode(13, OUTPUT);
//  digitalWrite(13, HIGH);
//
//  pinMode(8, OUTPUT);
//  digitalWrite(8, LOW);
//
//  Bridge.begin();
//  toggle.listenOnLocalhost();
//  toggle.begin();
//  Console.begin();
  // Bridge startup from AR02
  String rc(server);
  serverStr = rc;

  pinMode(13, OUTPUT);
  digitalWrite(13, LOW);
  Bridge.begin();
  digitalWrite(13, HIGH);

  pinMode(8, OUTPUT);          
  
  digitalWrite(8, HIGH);
  
  Console.begin();
}

void loop() 
{ 
   currentMillis = millis();
   if (currentMillis - lastMillis > interval)
   { 
      lastMillis = currentMillis;
      current = calculateCurrent();
      String c = String(current,5);
      String i = "AR12"; 
      data = "curr1=" + c + "&ardID=" + i;
      data = c;
      Console.println(c + " A");
      
      if (client.connect(server, 5000))
      {
          Console.println("CONNECTED!");

          // POST request to server
          client.println("POST /postData HTTP/1.1");
          client.println("Host: 127.0.0.1:5000");
          client.println("Content-Type: text/plain");
          client.println("Content-Length: 7");
          client.println("Connection: Keep-Alive");
          client.println();
          client.println(data);
       }
       else
       {
          Console.println("Not connected!");
       }

       // Initialize the client library
      HttpClient client;
    
      // Make a GET HTTP request:
      client.get("http://" + serverStr + ":5000/getStatus");
          
      // if there are incoming bytes available
      // from the server, read them and print them:
      while (client.available() && command.length() <= 3) {
        char c = client.read();
        String rc(c);

        command += rc;
      }

        //prints command from server
        Console.println(command);

        // turns outlet off
        if (command == "off"){
          Console.println("Turned off");

          // turns pin 13 & 8 off along with the rest
          digitalWrite(13, LOW);     
          digitalWrite(12, LOW);
          digitalWrite(11, LOW);     
          digitalWrite(10, LOW);
          digitalWrite(9, LOW);     
          digitalWrite(8, LOW);
          digitalWrite(7, LOW);     
          digitalWrite(6, LOW);
          digitalWrite(5, LOW);
          digitalWrite(4, LOW);     
          digitalWrite(3, LOW);
          digitalWrite(2, LOW);     
          digitalWrite(1, LOW);
        }
        
        // turns outlet on
        else if (command == "on"){
          Console.println("Turned on");

          // turns pin 13 & 8 on
          digitalWrite(13, HIGH);
          digitalWrite(8, HIGH);
        }

        command = "";
        Console.flush();
       }

}

double calculateCurrent() 
{
  voltage = getVPP();
  VRMS = (voltage/2.0) *0.707; 
  ampsRMS = (VRMS * 1000)/mVperAmp;
  
  /*Console.print(ampsRMS);
  Console.println(" Amps RMS");*/

  return ampsRMS;
}

double getVPP() 
{
  double result;
  
  int readValue;          // value read from the sensor
  int maxValue = 0;       // store max value here
  int minValue = 1024;    // store min value here
  
  uint32_t start_time = millis();
  while((millis()-start_time) < 1000) //sample for 1 Sec 
  { 
    readValue = analogRead(A0);
    // see if you have a new maxValue
    if (readValue > maxValue) 
    {
       /*record the maximum sensor value*/
       maxValue = readValue;
    }
    if (readValue < minValue) 
    {
      /*record the maximum sensor value*/
      minValue = readValue;
    }
  }
   
  // Subtract min from max
  result = ((maxValue - minValue) * 5.0)/1024.0;
      
  return result;
}
