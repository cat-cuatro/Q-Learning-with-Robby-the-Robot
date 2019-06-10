#include <iostream>
#include <math.h>
#include <time.h>
#include <cstdlib>
#include <iomanip>
using namespace std;
// Const identifiers
#define MAX_DIMENSION 10 // Working with a 10x10 grid
#define MIN_DIMENSION 0
#define MAX_QMATRIX 10000 // Wasn't sure how many extra states I'd encounter . . .
#define START_X 0 
#define START_Y 0
#define SENSORS 5 // each robot can sense in 5 locations: north, east, south, west, current location
#define TRAINING 0
#define TESTING 1
#define FALSE -1
/*** The following const identifiers are used for the grid model ***/
#define WALL 2
#define CAN 1
#define EMPTY 0
#define CAN_ROBBY 3
#define JUST_ROBBY 4
#define CAN_EARL 5
#define JUST_EARL 6
#define EARL_ROBBY_CAN 7
#define JUST_EARL_ROBBY 8
/*** Modifying anything above this line will require major changes to the program ***/

#define CAN_PLACEMENT_PERCENT 50
#define LEARNING_RATE 0.2
#define DISCOUNT_FACTOR 0.9
#define MAX_STEPS 200
#define CAPTURE_REWARD 10
#define FAILED_CAPTURE_REWARD -10

enum exploration{stage1 = 100, stage2 = 10, stage3 = 5, stage4 = 5, stage5 = 0, testStage = 10};
// these represent percentage values that robby will explore versus choose optimally.
enum actions{goNorth, goEast, goSouth, goWest, Collect = 4, Current = Collect, Capture = 4};
// these are to represent actions or state spaces. goNorth = 1, goEast = 2, etc.. unless defined in the enumerator as something else.
struct roboGrid{
  int cansPlaced; // number of cans placed in the grid
  int grid[MAX_DIMENSION][MAX_DIMENSION]; // maximum size of grid
  
};
struct robot{
  int cansCollected; // number of cans collected
  int row; // coordinates of robby
  int column;
  int totalReward; 
  int iterations; // number of steps Robby or Earl have done
};

struct qmatrix{ // state string encoded as NESWC (North, east, south, west, current)
  int stateString[SENSORS]; // holds current percept state
  double weights[SENSORS]; // holds weights associated with each action
  int index; // This is the index where this set of state, action pairs is stored in the qtable
};

// New structure:
struct robotMoveSet{
  qmatrix previous;
  qmatrix current;
  int actionPerformed;
  int oldValue; 
};
// If Robby is caught, it is absolutely necessary that I know his previous two states, and
// the action he performed that transitioned him to his current state. This is because when he's
// caught, I need to backtrace his operations by one step so that I can update his q-table to reflect
// that the decision he just made resulted in him being caught.

/*************************************/
// Robby/Earl Functions              //
/*************************************/
void initializeEarl(robot & earl, roboGrid & environment);
// initialize Earl to a desirable state
int performCapture(robot & earl, roboGrid & environment, int action, bool & captured, robot & robby); // called by: PerformAction()
// method specific to earl that performs edge case checking and if a capture was successful
int run_earl(robot & earl, robot & robby, roboGrid & environment, qmatrix ** robbygrid, qmatrix ** earlgrid, qmatrix & currRobby, qmatrix & currEarl, int flag); // yikes! Called by train_earl_robby(), test_earl_robby()
int test_earl_robby();
// Testing function earl & robby. This function does NOT change either robot's qtables.
int train_earl_robby(robot & robby, robot & earl, roboGrid & environment, qmatrix ** robbygrid, qmatrix ** earlgrid, int episodes);
// Training function for earl & robby. This function *does* change qtables.
int earlAction(robot & testBot, roboGrid & environment, int action, bool & captured, robot & robby);
// This function is a slightly different version of performAction(), and serves to produce desirable output for earl's specific nature.
// Returns a reward, and is called by earlChooseAction().
int earlChooseAction(robot & testBot, qmatrix & currentState, qmatrix ** actiongrid, roboGrid & environment, int flag, bool & captured, robot & robby);
// This function is a slightly different version of chooseAction(). Uses epsilon greedy decisionmaking to choose an action, and then calls
// earlAction() to execute that decision. Returns an action taken.
int scoutForRobby(robot & earl, roboGrid & environment, robot & robby);
// This is a small incentive function called by earlAction(). If Robby is within earl's percepts, earl is rewarded.
void updateStateSet(robotMoveSet & previousMoveSet, qmatrix & currentState, qmatrix & nextState, int action, qmatrix ** robbygrid);
// This is used to keep the backtrace for Robby's last state pair recent, so if he's captured, his qtable can be adjusted.
void fixQValues(robotMoveSet & moveSet, qmatrix ** robbygrid);
// Self descriptive name. Called when Robby is captured, and updates his qtable to reflect that.

/***			Robby Functions Below				***/

int performAction(robot & robby, roboGrid & environment, int action, bool & captured); // called by: chooseAction()
// perform an action, specified by enumerator data type. returns a reward.
void observe(robot & robby, roboGrid & environment, qmatrix & currentState);
// observes the environment and returns the state that robby observes.
int chooseAction(robot & robby, qmatrix & currentState, qmatrix ** actiongrid, roboGrid & environment, int flag, bool & captured);
// chooses an action based on weights in the qmatrix // called by: run_robby() and run_earl() . . (see earl.cpp)
int train_robby(robot & robby, roboGrid & environment, qmatrix ** actiongrid, int episodes);
// trains Robby some number of episodes
int test_robby(robot & robby, roboGrid & environment, qmatrix ** actiongrid, int episodes);
// tests robby, and does not modify qtable.
int run_robby(robot & robby, roboGrid & environment, qmatrix ** actiongrid, qmatrix & currentState, int flag);
// a vessel for Robby's operation within a single episode. Runs for the maximum number of steps,
// or until he collects all cans.
void resetRobby(robot & robby, roboGrid & environment);
// reset robby and prepare him for another episode.
/*************************************/
// State Specific Functions          //
/*************************************/
void refreshGrid(roboGrid & environment);
// create a new random environment for Robby.
int retrieveNextQval(qmatrix nextState);
// retrieves perceived max q value from the next state's reward list
void initEnvironment(robot & robby, roboGrid & environment);
// initialize environment for Robby.
void deallocateQmatrix(qmatrix ** actiongrid);
// deletes qmatrix array
int lookup(qmatrix & currentState, qmatrix ** actiongrid);
// search q matrix for an action, if it exists. returns -1 on fail.
void copyState(qmatrix & source, qmatrix & destination);
// copies data from source to destination
int addState(qmatrix & currentState, qmatrix ** actiongrid);
// adds the current state to the action lookup table
void printState(qmatrix & toPrint);
// given a state, prints its contents.
void displayQmatrix(qmatrix ** actiongrid);
// display entire action grid
void initializeQmatrix(qmatrix ** actiongrid);
/*** Added after Earl update ***/
// When Earl was added, it became necessary to monitor on a model where both agents were at all times, and so 
// these functions are used to do that. They use the constant definitions above to update in accordance to whatever
// action and agent decides to take.
void manageEnvironment(roboGrid & environment, int prevRow, int prevCol, int dirMoved, int flag, robot & testBot);
// Wrapper function that calls both newStateFourCaseCheck() and fourCaseCheck().
void newStateFourCaseCheck(roboGrid & environment, int row, int col, int flag);
void fourCaseCheck(roboGrid & environment, int prevRow, int prevCol, int flag);
// the above two functions do exactly what they sound like they do. They have a four case check, where when either
// Earl or Robby moves somewhere in the state space, it requires the model to be changed to reflect those movements.
/*************************************/
// Miscellaneous/Debugging Functions //
/*************************************/
void menuPrompt();
// prints menu in main.cpp
void displayObservation(int stateString[]);
// prints stateString
void printarena(roboGrid & arena);
// prints grid state
void dumpRobbyData(robot & robby);
// prints robby's data
void printAction(int action);
// prints an action map
/*** Added after Earl update ***/
int q_update(int & currIndex, int action, int reward, qmatrix & nextState, qmatrix ** actiongrid, int lookup);
// it became desirable to place the q-learning algorithm inside of a function to reduce clutter in an already 
// massive run() function. (See run_earl())
void prettyPrintArena(roboGrid & arena, robot & earl, robot & robby);
// pretty display for the arena space.
