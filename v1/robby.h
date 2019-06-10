#include <iostream>
#include <math.h>
#include <time.h>
#include <cstdlib>
#include <iomanip>
using namespace std;
// Const identifiers
#define MAX_DIMENSION 10
#define MIN_DIMENSION 0
#define CAN_PLACEMENT_PERCENT 50
#define MAX_QMATRIX 10000
#define START_X 0
#define START_Y 0
#define SENSORS 5
#define WALL 2
#define CAN 1
#define EMPTY 0
#define CAN_ROBBY 3
#define JUST_ROBBY 4
#define CAN_EARL 5
#define JUST_EARL 6
#define EARL_ROBBY_CAN 7
#define JUST_EARL_ROBBY 8
#define LEARNING_RATE 0.2
#define DISCOUNT_FACTOR 0.9
#define MAX_STEPS 200
#define TRAINING 0
#define TESTING 1
#define FALSE -1
#define CAPTURE_REWARD 10
#define FAILED_CAPTURE_REWARD -10
enum exploration{stage1 = 100, stage2 = 10, stage3 = 5, stage4 = 5, stage5 = 0, testStage = 10};
// these represent percentage values that robby will explore versus choose optimally.
enum actions{goNorth, goEast, goSouth, goWest, Collect = 4, Current = Collect, Capture = 4};
// these are to represent actions or state spaces. goNorth = 1, goEast = 2, etc.. unless defined in the enumerator.
struct roboGrid{
  int cansPlaced; // number of cans placed in the grid
  int grid[MAX_DIMENSION][MAX_DIMENSION]; // maximum size of grid
  
};
struct robot{
  int cansCollected; // number of cans collected
  int row; // coordinates of robby
  int column;
  int totalReward;
  int iterations;
};

struct qmatrix{ // state string encoded as NESWC (North, east, south, west, current)
  int stateString[SENSORS]; // holds current percept state
  double weights[SENSORS]; // holds weights associated with each action
  int index;
};

struct robotMoveSet{
  qmatrix previous;
  qmatrix current;
  int actionPerformed;
  int oldValue;
  
};
/*************************************/
// Robby/Earl Functions              //
/*************************************/
void initializeEarl(robot & earl, roboGrid & environment);
// initialize Earl to a desirable state
int performCapture(robot & earl, roboGrid & environment, int action, bool & captured, robot & robby); // called by: PerformAction()
// method specific to earl that performs edge case checking and if a capture was successful
int run_earl(robot & earl, robot & robby, roboGrid & environment, qmatrix ** robbygrid, qmatrix ** earlgrid, qmatrix & currRobby, qmatrix & currEarl, int flag); // yikes! Called by train_earl_robby(), test_earl_robby()
int test_earl_robby();
int train_earl_robby(robot & robby, robot & earl, roboGrid & environment, qmatrix ** robbygrid, qmatrix ** earlgrid, int episodes);
int earlAction(robot & testBot, roboGrid & environment, int action, bool & captured, robot & robby);
int earlChooseAction(robot & testBot, qmatrix & currentState, qmatrix ** actiongrid, roboGrid & environment, int flag, bool & captured, robot & robby);
int scoutForRobby(robot & earl, roboGrid & environment, robot & robby);
void updateStateSet(robotMoveSet & previousMoveSet, qmatrix & currentState, qmatrix & nextState, int action, qmatrix ** robbygrid);
void fixQValues(robotMoveSet & moveSet, qmatrix ** robbygrid);
// ROBBY BELOW *********************************************************

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
void manageEnvironment(roboGrid & environment, int prevRow, int prevCol, int dirMoved, int flag, robot & testBot);
void newStateFourCaseCheck(roboGrid & environment, int row, int col, int flag);
void fourCaseCheck(roboGrid & environment, int prevRow, int prevCol, int flag);
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
int q_update(int & currIndex, int action, int reward, qmatrix & nextState, qmatrix ** actiongrid, int lookup);
