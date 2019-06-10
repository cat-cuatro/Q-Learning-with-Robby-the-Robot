// John Lorenz IV // CS 441 // Programming HW 3
// File contains all function implementations that are
// specific to the Q-Learning algorithm and state space
#include "robby.h"
/*****************************/
/* Robby the Robot Functions */
/*****************************/
int chooseAction(robot & testBot, qmatrix & currentState, qmatrix ** actiongrid, roboGrid & environment, int flag, bool & captured){
  int action = -1;
  // first I must find the optimal choice
  int index = 0;
  int rewardReceived = 0;
  int actionWeightPairs[SENSORS][2];// pairs each action with its respective weight
  double bestPair[2];
  int i = 0;
  int chance = 0; // chance of making a non-optimal choice
  int dice = rand() % 101;
  index = lookup(currentState, actiongrid);
  if(index == -1){
    index = addState(currentState, actiongrid);
  }
  // by this point, the state is guaranteed to have been initialized, or found.
  // Therefore, it is safe to access it.
  bestPair[0] = 0;
  bestPair[1] = actiongrid[index]->weights[0]; // Initializing pair value to 0 before iterating loop
  for(i = 0; i < SENSORS; ++i){ // assign the weights to their respective actions
    actionWeightPairs[i][1] = actiongrid[index]->weights[i];
    if(bestPair[1] <= actiongrid[index]->weights[i]){
      bestPair[1] = actiongrid[index]->weights[i];
      bestPair[0] = i;
    }
  }
  // now testBot can make an informed choice using epsilon greedy.
  if(testBot.iterations >= 50 && testBot.iterations <= 100){
    chance = stage2; // see enumerator "exploration" in robby.h for documentation
  }
  else if(testBot.iterations > 100 && testBot.iterations <= 150){
    chance = stage3;
  }
  else if(testBot.iterations > 150 && testBot.iterations < 200){
    chance = stage4;
  }
  else if(testBot.iterations >= 200){
    chance = stage5;
  }
  else{ // if testBot.iterations <= 499
    chance = stage1;
  }
  // now I choose an action based on the chances
  if(flag == TESTING){
    chance = testStage; // if I'm not training testBot, set epsilon to constant 10%
  }
  if(dice > chance){ // then perform optimal action
    rewardReceived = performAction(testBot, environment, bestPair[0], captured);
    action = bestPair[0];
  }
  else{ // otherwise, perform randomly.
    action = rand() % 5;
    rewardReceived = performAction(testBot, environment, action, captured);
  }
  return action;
}
void printAction(int action){ // debugging function.
  switch(action){
    case 0:
      cout << "Going north .. " << endl;
      break;
    case 1:
      cout << "Going east .. " << endl;
      break;
    case 2:
      cout << "Going south .. " << endl;
      break;
    case 3:
      cout << "Going west .. " << endl;
      break;
    case 4:
      cout << "Collecting .. " << endl;
      break;
    default:
      cout << "Action code " << action << " not valid." << endl;
      break;
  }
}
int test_robby(robot & robby, roboGrid & environment, qmatrix ** actiongrid, int episodes){
  int cansCollected = 0;
  int i = 0;
  int flag = TESTING;
  robot oldRobby;
  oldRobby.totalReward = 0;
  oldRobby.cansCollected = 0;
  qmatrix currentState; // I need to initialize the current state of Robby
  int index = 0;        // to 'jumpstart' the algorithm
  float avgReward = 0;
  float avgCans = 0;
  while(i < episodes){ // run for some period of time
    observe(robby, environment, currentState); // Observe where I am
    index = lookup(currentState, actiongrid); // check to see if this data exists. If not, we must add it before running Robby.
    if(index == -1){ // then I need to add this observed state to the grid.
      index = addState(currentState, actiongrid); // grab the index to make finding state informatione easy.
    }
    currentState.index = index;
    cansCollected = run_robby(robby, environment, actiongrid, currentState, flag) + cansCollected;
    ++i;
    oldRobby.cansCollected = robby.cansCollected + oldRobby.cansCollected;
    oldRobby.totalReward = robby.totalReward + oldRobby.totalReward;
    if(i % 100 == 0){
      cout << "Averages for " << (i-100) << " to " << i << endl;
      avgReward = (float)oldRobby.totalReward/100.0; // type convert integer to float for floating point arithmetic
      avgCans = (float)oldRobby.cansCollected/100.0;
      cout << "Avg. Reward: " << avgReward << " Avg. Cans: " << avgCans << endl;
      oldRobby.cansCollected = 0;
      oldRobby.totalReward = 0; 
    }
    resetRobby(robby, environment);
    refreshGrid(environment);
  }
  return cansCollected;
}

int train_robby(robot & robby, roboGrid & environment, qmatrix ** actiongrid, int episodes){
  int cansCollected = 0;
  int i = 0;
  int flag = TRAINING;
  robot oldRobby;
  oldRobby.totalReward = 0;
  oldRobby.cansCollected = 0;
  qmatrix currentState;
  int index = 0;
  float avgReward = 0;
  float avgCans = 0;
  bool captured = false;
  while(i < episodes){ // see comments above. Functions are virtually the same, but feed a different flag to the run_robby method.
    observe(robby, environment, currentState);
    index = lookup(currentState, actiongrid);
    if(index == -1){ // then I need to add this observed state to the grid.
      index = addState(currentState, actiongrid);
    }
    currentState.index = index;
    cansCollected = run_robby(robby, environment, actiongrid, currentState, flag) + cansCollected;
    ++i;
    oldRobby.cansCollected = robby.cansCollected + oldRobby.cansCollected;
    oldRobby.totalReward = robby.totalReward + oldRobby.totalReward;
    if(i % 100 == 0){
      cout << "Averages for " << (i-100) << " to " << i << endl;
      avgReward = (float)oldRobby.totalReward/100.0;
      avgCans = (float)oldRobby.cansCollected/100.0;
      cout << "Avg. Reward: " << avgReward << " Avg. Cans: " << avgCans << endl;
      oldRobby.cansCollected = 0;
      oldRobby.totalReward = 0;
      
    }

    resetRobby(robby, environment);
    refreshGrid(environment);
  }
  cout << "Training completed. Robby ran " << i << " episodes." << endl;
  return cansCollected;
}
int run_robby(robot & robby, roboGrid & environment, qmatrix ** actiongrid, qmatrix & currentState, int flag){
  int i = 0;
  qmatrix nextState;
  int indexLookup = 0;
  int rewardDiff = 0;
  int prevReward = robby.totalReward;
  int actionTaken = 0;
  int maxQ = 0;
  bool captured = false;
  while(i < MAX_STEPS){
    observe(robby, environment, currentState); // loads observed state into currentState object
    actionTaken = chooseAction(robby, currentState, actiongrid, environment, flag, captured); // chooses and performs an action using current state
    observe(robby, environment, nextState);    //loads observes NEW state into nextState object
    indexLookup = lookup(nextState, actiongrid); // check to see if the newly observed state exists in Qmatrix
    nextState.index = indexLookup;
    rewardDiff = robby.totalReward - prevReward; // calculate the difference to see if Robby received a reward (pos or negative)
    prevReward = robby.totalReward; // update reward tally for robby for next step
    if(indexLookup == -1){// if -1, then state weights must be 0 for the NEW state (as it doesn't exist!)
      // so update accordingly
      if(flag == TRAINING){ // only update weights if training
        actiongrid[currentState.index]->weights[actionTaken]= actiongrid[currentState.index]->weights[actionTaken]+
        LEARNING_RATE*(rewardDiff + DISCOUNT_FACTOR*0-actiongrid[currentState.index]->weights[actionTaken]);
        nextState.index = addState(nextState, actiongrid);
      }
    }
    else{ // if index already exists, then state weights exist, so use those to update state action pair
      if(flag == TRAINING){
        maxQ = retrieveNextQval(nextState);
        actiongrid[currentState.index]->weights[actionTaken]= actiongrid[currentState.index]->weights[actionTaken]+
        LEARNING_RATE*(rewardDiff + (DISCOUNT_FACTOR*maxQ)-
        actiongrid[currentState.index]->weights[actionTaken]);
      }
    }
    // now that the weights are updated, and I've performed an action, increment counter
    copyState(nextState, currentState); // copy(src, dst);
    currentState.index = nextState.index; // copy index where this state exists in the qtable

    ++i;
    ++robby.iterations;
  }
  if(environment.cansPlaced == 0){
  }
  return robby.cansCollected;
}
// Fills the currentState object with a percept sequence that relates to the robot's current x and y coordinates.
void observe(robot & testBot, roboGrid & environment, qmatrix & currentState){ 
  int currRow = testBot.row;                                                   
  int currCol = testBot.column;
  int stateString[SENSORS];

  if(currRow-1 < MIN_DIMENSION){ // check north
    stateString[goNorth] = WALL;
  }
  else{
    stateString[goNorth] = environment.grid[currRow-1][currCol];
  }

  if(currCol+1 >= MAX_DIMENSION){ // check east
    stateString[goEast] = WALL;
  }
  else{
    stateString[goEast] = environment.grid[currRow][currCol+1];
  }

  if(currRow+1 >= MAX_DIMENSION){ // check south
    stateString[goSouth] = WALL;
  }
  else{
    stateString[goSouth] = environment.grid[currRow+1][currCol];
  }

  if(currCol-1 < MIN_DIMENSION){ // check west
    stateString[goWest] = WALL;
  }
  else{
    stateString[goWest] = environment.grid[currRow][currCol-1];
  }

  // assert current
  stateString[Current] = environment.grid[testBot.row][testBot.column];
  for(int i = 0; i < SENSORS; ++i){
    currentState.stateString[i] = stateString[i]; // assign observations
    currentState.weights[i] = 0; // initialize weights to 0.
  }
}
int performAction(robot & testBot, roboGrid & environment, int action, bool & captured){
  int reward = 0;
  int previousRow = testBot.row;
  int previousColumn = testBot.column;
  int directionMoved = 0;
  switch(action){
    case goNorth:
      if(testBot.row == MIN_DIMENSION){ // if robby crashes into a wall
        reward = -5;
      }
      else{ // otherwise, robby moves north
        --testBot.row;
        directionMoved = 1;
      }
      break;
    case goEast:
      if(testBot.column == (MAX_DIMENSION-1)){ // if robby crashes into a wall
        reward = -5;
      }
      else{
        ++testBot.column; // otherwise, robby moves east
        directionMoved = 2;
      }
      break;
    case goSouth:
      if(testBot.row == (MAX_DIMENSION-1)){ // if robby crashes into a wall
        reward = -5;
      }
      else{
        ++testBot.row; // otherwise, robby moves south
        directionMoved = 3;
      }
      break;
    case goWest:
      if(testBot.column == MIN_DIMENSION){ // if robby crashes into a wall
        reward = -5;
      }
      else{
        --testBot.column; // otherwise, robby moves west.
        directionMoved = 4;
      }
      break;
    case Collect:
      if(environment.grid[testBot.row][testBot.column] == CAN_ROBBY){ // if there is a can, reward robby!
        reward = 10;
        environment.grid[testBot.row][testBot.column] = JUST_ROBBY; // collect can
        --environment.cansPlaced;
        ++testBot.cansCollected;
      }
      else if(environment.grid[testBot.row][testBot.column] == EARL_ROBBY_CAN){
        reward = 10;
        environment.grid[testBot.row][testBot.column] = JUST_EARL_ROBBY; // collect can
        --environment.cansPlaced;
        ++testBot.cansCollected;
      }
      else if(environment.grid[testBot.row][testBot.column] == JUST_EARL_ROBBY){
        reward = -1;
      }
      else{
        reward = -1; // otherwise, discipline robby.
      }
      break;
    default: // handle corrupt input
      cout << "Error! The robot cannot perform an action of this parameter!" << endl;
      break;
  }
  manageEnvironment(environment, previousRow, previousColumn, directionMoved, 0, testBot); // 0 indicates this is Robby entering
  testBot.totalReward = testBot.totalReward + reward;
  return reward;
}
void resetRobby(robot & robby, roboGrid & environment){ // set robby back to initial state
  robby.cansCollected = 0;
  robby.row = 0;
  robby.column = 0;
  robby.totalReward = 0;
  robby.iterations = 0;
  environment.grid[robby.row][robby.column] = JUST_ROBBY;
}
/****************************/
/* State Specific Functions */
/****************************/
int retrieveNextQval(qmatrix nextState){
  int maximumVal = nextState.weights[0];
  int i = 0;
  for(i = 0; i < SENSORS; ++i){
    if(maximumVal < nextState.weights[i]){
      maximumVal = nextState.weights[i];
    }
  }
  return maximumVal;
}
void refreshGrid(roboGrid & environment){ // set environment back to initial state
  int dice = 0;
  int i = 0;
  int j = 0;
  environment.cansPlaced = 0;
  for(i = 0; i < MAX_DIMENSION; ++i){
    for(j = 0; j < MAX_DIMENSION; ++j){
      environment.grid[i][j] = 0; // initialize all grid values to 0
    }
  }
  for(i = 0; i < MAX_DIMENSION; ++i){
    for(j = 0; j < MAX_DIMENSION; ++j){
      dice = rand() % 100;
      if(dice >= CAN_PLACEMENT_PERCENT){
        ++environment.cansPlaced;
        environment.grid[i][j] = 1; // with a 50% chance, place a can on the grid.
      }
    }
  }
  if(environment.grid[MIN_DIMENSION][MIN_DIMENSION] == 1){
    --environment.cansPlaced;
    environment.grid[MIN_DIMENSION][MIN_DIMENSION] = 0; // [0][0] is the same as min dimension
  }
  if(environment.grid[MAX_DIMENSION-1][MAX_DIMENSION-1] == 1){
    --environment.cansPlaced;
    environment.grid[MAX_DIMENSION-1][MAX_DIMENSION-1] = 0;
  }
}
void initEnvironment(robot & robby, roboGrid & environment){ // set environment to an initial state
  int dice = 0;
  int i = 0;
  int j = 0;
  robby.cansCollected = 0;
  robby.row = rand() % MAX_DIMENSION;
  robby.column = rand() % MAX_DIMENSION;
  robby.totalReward = 0;
  robby.iterations = 0;
  environment.cansPlaced = 0;
  for(i = 0; i < MAX_DIMENSION; ++i){
    for(j = 0; j < MAX_DIMENSION; ++j){
      dice = rand() % 100;
      if(dice >= CAN_PLACEMENT_PERCENT){
        ++environment.cansPlaced;
        environment.grid[i][j] = 1;
      }
      else{
        environment.grid[i][j] = 0;
      }
    }
  }
  if(environment.grid[START_X][START_Y] == 1){ // remove can from 0, 0
    --environment.cansPlaced;
    environment.grid[START_X][START_Y] = 0;
  }
  if(environment.grid[MAX_DIMENSION-1][MAX_DIMENSION-1] == 1){ // remove can from 9, 9
    --environment.cansPlaced;
    environment.grid[MAX_DIMENSION-1][MAX_DIMENSION-1] = 0;
  }
}
void printState(qmatrix & toPrint){
  for(int i = 0; i < SENSORS; ++i){
    cout << toPrint.stateString[i] << "\t";
  }
  for(int i = 0; i < SENSORS; ++i){
    cout << fixed << setprecision(2) << toPrint.weights[i] << ", ";
  }
  cout << endl;
}
int addState(qmatrix & currentState, qmatrix ** actiongrid){
  int index = -1;
  int i = 0;
  int j = 0;
  while(actiongrid[i] != NULL){
    ++i;
  }
  index = i;
  if(i >= MAX_QMATRIX){
    cout << "You've reached the maximum number of states that can be stored!" << endl;
    return -1;
  }
  actiongrid[i] = new qmatrix; // create new object in qmatrix
  copyState(currentState, (*actiongrid[i])); // copy state data into structure
  currentState.index = index; // update indices 
  actiongrid[i]->index = index;
  return index;
}
void copyState(qmatrix & source, qmatrix & destination){
  int i = 0;
  for(i = 0; i < SENSORS; ++i){
    destination.stateString[i] = source.stateString[i];
    destination.weights[i] = source.weights[i];
  }
  destination.index = source.index;
}
void deallocateQmatrix(qmatrix ** actiongrid){
  int i = 0;
  for(i = 0; i < MAX_QMATRIX; ++i){
    if(actiongrid[i] != NULL){
      delete actiongrid[i];
    }
  }
}
int lookup(qmatrix & currentState, qmatrix ** actiongrid){ // search qmatrix for a specific state.
  int index = -1;
  int i = 0;
  int j = 0;
  int k = 0;
  bool flag = true;
  for(i = 0; i < MAX_QMATRIX; ++i){
    flag = true;
    if(actiongrid[i] == NULL){
      return index;
    }
    for(j = 0; j < SENSORS; ++j){
      if(actiongrid[i]->stateString[j] == currentState.stateString[j]){
        // do nothing, as we are still true
      }
      else{
        flag = false;
        j = SENSORS; // break loop we are not a match
      }
    }
    if(flag == true){ // then state exists in list, assign return value to index location
      index = i;
      for(k = 0; k < SENSORS; ++k){
        currentState.weights[k] = actiongrid[index]->weights[k];
      }
      i = MAX_QMATRIX; // break loop
    }
  }
  return index;
}
void displayQmatrix(qmatrix ** actiongrid){
  int i = 0;
  cout << "North\tEast\tSouth\tWest\tCollect" << endl;
  while(actiongrid[i] != NULL){
    printState((*actiongrid[i])); // dereference, and print, not sure if this will work
    ++i;
  }
}
void initializeQmatrix(qmatrix ** actiongrid){
  for(int i = 0; i < MAX_QMATRIX; ++i){
    actiongrid[i] = NULL;
  }
}
/***********************/
// Debugging Functions //
/***********************/
void displayObservation(int stateString[]){
  int i = 0;
  for(i = 0; i < SENSORS; ++i){
    cout << stateString[i] << " ";
  }
  cout << endl;
}
void printarena(roboGrid & arena){
  int i = 0;
  int j = 0;
  cout << "Cans on this grid: " << arena.cansPlaced << endl;
  for(i = 0; i < MAX_DIMENSION; ++i){
    for(j = 0; j < MAX_DIMENSION; ++j){
      cout << arena.grid[i][j] << " ";
    }
    cout << endl;
  }
}
void dumpRobbyData(robot & robby){
  cout << "Robby's data: " << endl;
  cout << "Cans: " << robby.cansCollected << " Row: " << robby.row << " Column: " << robby.column << " Reward: " << robby.totalReward << endl;
}

/*** SOME EARL FUNCTIONS  ***/
// The reason why these Earl functions exist in robby.cpp is because they borrow code logic
// from Robby's sequences.
int earlAction(robot & testBot, roboGrid & environment, int action, bool & captured, robot & robby){
  int reward = 0;
  int prevRow = testBot.row;
  int prevCol = testBot.column;
  int dirMoved = 0;
  switch(action){
    case goNorth:
      if(testBot.row == MIN_DIMENSION){ // if earl crashes into a wall
        reward = -5;
      }
      else{ // otherwise, earl moves north
        --testBot.row;
      }
      break;
    case goEast:
      if(testBot.column == (MAX_DIMENSION-1)){ // if earl crashes into a wall
        reward = -5;
      }
      else{
        ++testBot.column; // otherwise, eatl moves east
      }
      break;
    case goSouth:
      if(testBot.row == (MAX_DIMENSION-1)){ // if earl crashes into a wall
        reward = -5;
      }
      else{
        ++testBot.row; // otherwise, earl moves south
      }
      break;
    case goWest:
      if(testBot.column == MIN_DIMENSION){ // if earl crashes into a wall
        reward = -5;
      }
      else{
        --testBot.column; // otherwise, earl moves west.
      }
      break;
    case Capture:
      reward = performCapture(testBot, environment, action, captured, robby);
      break;
    default: // handle corrupt input
      cout << "Error! The robot cannot perform an action of this parameter!" << endl;
      break;
  }
  // I want to incentivize earl to have Robby in his visual, and so I should reward him if Robby is.
  manageEnvironment(environment, prevRow, prevCol, dirMoved, 1, testBot); // 1 indicates this is Earl entering the function
  reward = reward + scoutForRobby(testBot, environment, robby); // if I see Robby in this new state, then it is desirable.
  testBot.totalReward = testBot.totalReward + reward;
  return reward;
}
int earlChooseAction(robot & testBot, qmatrix & currentState, qmatrix ** actiongrid, roboGrid & environment, int flag, bool & captured, robot & robby){
  int action = -1;
  // first I must find the optimal choice
  int index = 0;
  int rewardReceived = 0;
  int actionWeightPairs[SENSORS][2];// pairs each action with its respective weight
  double bestPair[2];
  int i = 0;
  int chance = 0; // chance of making a non-optimal choice
  int dice = rand() % 101;
  index = lookup(currentState, actiongrid);
  if(index == -1){
    index = addState(currentState, actiongrid);
  }
  // by this point, the state is guaranteed to have been initialized, or found.
  // Therefore, it is safe to access it.
  bestPair[0] = 0;
  bestPair[1] = actiongrid[index]->weights[0]; // Initializing pair value to 0 before iterating loop
  for(i = 0; i < SENSORS; ++i){ // assign the weights to their respective actions
    actionWeightPairs[i][1] = actiongrid[index]->weights[i];
    if(bestPair[1] <= actiongrid[index]->weights[i]){
      bestPair[1] = actiongrid[index]->weights[i];
      bestPair[0] = i;
    }
  }
  // now testBot can make an informed choice using epsilon greedy.
  if(testBot.iterations >= 50 && testBot.iterations <= 100){
    chance = stage2; // see enumerator "exploration" in robby.h for documentation
  }
  else if(testBot.iterations > 100 && testBot.iterations <= 150){
    chance = stage3;
  }
  else if(testBot.iterations > 150 && testBot.iterations < 200){
    chance = stage4;
  }
  else if(testBot.iterations >= 200){
    chance = stage5;
  }
  else{ // if testBot.iterations <= 49
    chance = stage1;
  }
  // now I choose an action based on the chances
  if(flag == TESTING){
    chance = testStage; // if I'm not training testBot, set epsilon to constant 10%
  }
  if(dice > chance){ // then perform optimal action
    rewardReceived = earlAction(testBot, environment, bestPair[0], captured, robby);
    action = bestPair[0];
  }
  else{ // otherwise, perform randomly.
    action = rand() % 5;
    rewardReceived = earlAction(testBot, environment, action, captured, robby);
  }
  // now update weights
  return action;
}
void updateStateSet(robotMoveSet & previousMoveSet, qmatrix & currentState, qmatrix & nextState, int action, qmatrix ** robbygrid){
  copyState(currentState, previousMoveSet.previous);
  copyState(nextState, previousMoveSet.current);
  previousMoveSet.actionPerformed = action;
  if(robbygrid[currentState.index] != NULL){ // make sure the state we're referencing exists.
    previousMoveSet.oldValue = robbygrid[currentState.index]->weights[action];
  }
  else{ // if it doesn't, then the state's value would be 0.
    previousMoveSet.oldValue = 0;
  }
}
void fixQValues(robotMoveSet & moveSet, qmatrix ** robbygrid){
  int reward = -CAPTURE_REWARD; // Robby is penalized -100 if he is captured. 
  int reply = 0;
  robbygrid[moveSet.previous.index]->weights[moveSet.actionPerformed] =
  q_update(moveSet.previous.index, moveSet.actionPerformed, reward, moveSet.current, robbygrid, 1);
  // this line of code updates Robby's qtable to what it should be.
}
