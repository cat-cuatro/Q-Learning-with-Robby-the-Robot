#include "robby.h"
int train_earl_robby(robot & robby, robot & earl, roboGrid & environment, qmatrix ** robbygrid, qmatrix ** earlgrid, int episodes){
  int cansCollected = 0;
  int i = 0;
  int flag = TRAINING;
  robot oldRobby;
  oldRobby.totalReward = 0;
  oldRobby.cansCollected = 0;
  qmatrix robbyCurrentState;
  qmatrix earlCurrentState;
  int robbyIndex = 0;
  int earlIndex = 0;
  float avgReward = 0;
  float avgCans = 0;
  int robbyCaptures = 0;
  float earlReward = 0;
  float earlAvg = 0;
  cout << "Do you want to TRAIN these robots, or do you want to TEST these robots?" << endl;
  cout << "0 = train, 1 = test. Bad input = train" << endl;
  cin >> flag;
  cin.ignore(100,'\n');
  if(flag != 0 && flag != 1){
    cout << "Bad input. Defaulting to training." << endl;
    flag = TRAINING;
    cin.clear();
  }
  while(i < episodes){
    /*** Initialize current state for Robby ***/
    observe(robby, environment, robbyCurrentState); // <-- Analyze Robby's percepts
    robbyIndex = lookup(robbyCurrentState, robbygrid); // <-- Check if he's seen this state before.
    if(robbyIndex == -1){ // If not, now he has, so we should add it to his table.
      robbyIndex = addState(robbyCurrentState, robbygrid);
    }
    robbyCurrentState.index = robbyIndex; 
    /*** Now initialize current state for Earl ***/
    observe(earl, environment, earlCurrentState);   // <-- Analyze this robot's current percepts.
    earlIndex = lookup(earlCurrentState, earlgrid); // <-- Does this state exist?
    if(earlIndex == -1){ // If not, create it.
      earlIndex = addState(earlCurrentState, earlgrid);
    }
    earlCurrentState.index = earlIndex; // Keep track of this state's index for easier referencing.
    /*** Now that both robots are initialized, we can proceed with training ***/
    cansCollected = run_earl(earl, robby, environment, robbygrid, earlgrid, robbyCurrentState, earlCurrentState, flag);
    ++i; // Increment number of episodes performed.
    /*** Process can collection data. ***/
    oldRobby.cansCollected = robby.cansCollected + oldRobby.cansCollected;
    oldRobby.totalReward = robby.totalReward + oldRobby.totalReward;
    robbyCaptures = earl.cansCollected + robbyCaptures;
    earlReward = earl.totalReward + earlReward;
    if(i % 100 == 0){
      cout << "Averages for " << (i-100) << " to " << i << endl;
      avgReward = (float)oldRobby.totalReward/100.0;
      avgCans = (float)oldRobby.cansCollected/100.0;
      earlAvg = (float)earlReward/100.0;
      cout << "// Robby Data //" << endl;
      cout << "Avg. Reward: " << avgReward << " Avg. Cans: " << avgCans << endl;
      cout << "// Earl Data //" << endl;
      cout << "Robby was captured: " << robbyCaptures << " times." << endl;
      cout << "Avg. Reward: " << earlAvg << endl;
      oldRobby.cansCollected = 0;
      oldRobby.totalReward = 0;
      robbyCaptures = 0;
      earlReward = 0;
    }
    refreshGrid(environment);
    resetRobby(robby, environment);
    initializeEarl(earl, environment);
  }
  

}
int run_earl(robot & earl, robot & robby, roboGrid & environment, qmatrix ** robbygrid, qmatrix ** earlgrid, qmatrix & currRobby, qmatrix & currEarl, int flag){
  int i = 0;
  qmatrix earlNextState;
  qmatrix robbyNextState;
  robotMoveSet previousMoveSet;
  int indexLookup = 0;
  int rewardDiff = 0;
  int prevReward = 0;
  int actionTaken = 0;
  int maxQ = 0;
  int earlPrevReward = 0;
  bool captured = false;
  while(i < MAX_STEPS && captured == false){
  /*** First, we want Robby to take a turn on the grid ***/
    observe(robby, environment, currRobby);
    // Robby must first observe his environment, and load his percepts
    actionTaken = chooseAction(robby, currRobby, robbygrid, environment, flag, captured);
    // Robby can now make a decision based on what he sees.
    observe(robby, environment, robbyNextState);
    // Now Robby will want to observe the state he has transitioned into.
    // Check to see if Robby's next state exists in his q-table. If it doesn't, indexLookup will be -1.
    indexLookup = lookup(robbyNextState, robbygrid);
    if(indexLookup == -1){
      indexLookup = addState(robbyNextState, robbygrid);
    }
    robbyNextState.index = indexLookup;
    // Calculate how much Robby has gained (or lost!) in terms of his reward.
    rewardDiff = robby.totalReward - prevReward;
    prevReward = robby.totalReward;
    // Update robby's q-table in accordance to what he's done
    if(flag == TRAINING){ // Don't track extra states if I'm not training the agents. Similarly, I don't want to update q-tables.
      updateStateSet(previousMoveSet, currRobby, robbyNextState, actionTaken, robbygrid); // <-- this tracks Robby's former action set.
      robbygrid[currRobby.index]->weights[actionTaken] = q_update(currRobby.index, actionTaken, rewardDiff, robbyNextState, robbygrid, indexLookup);
    }
    // Copy the next state as Robby's current state.
    copyState(robbyNextState, currRobby);
    // now, update robby's current index for the next step sequence.
    currRobby.index = robbyNextState.index;
    ++robby.iterations;
  /*** Next, we want Earl to take a turn on the grid ***/
    observe(earl, environment, currEarl); // This is the same exact sequence as above, but not commented.
    actionTaken = earlChooseAction(earl, currEarl, earlgrid, environment, flag, captured, robby);
    observe(earl, environment, earlNextState);
    indexLookup = lookup(earlNextState, earlgrid);
    earlNextState.index = indexLookup;
    rewardDiff = earl.totalReward - earlPrevReward;
    earlPrevReward = earl.totalReward;
    if(flag == TRAINING){ // don't update earl's qtable if I'm not training him
      earlgrid[currEarl.index]->weights[actionTaken] = q_update(currEarl.index, actionTaken, rewardDiff, earlNextState, earlgrid, indexLookup);
    }
    copyState(earlNextState, currEarl);
    currEarl.index = earlNextState.index;
  
    if(captured == true && flag == TRAINING){ // Robby has been captured! His q-value matrix must be updated to reflect his bad decision!
      // in order to simulate this effect, I must step back and modify how I updated his Q-Value.
      fixQValues(previousMoveSet, robbygrid);
    } 
    if(captured == true){
      ++earl.cansCollected; // actually just signifies that robby was caught.
    }
    ++i;
    ++earl.iterations;
  }
  return robby.cansCollected;
}
void initializeEarl(robot & earl, roboGrid & environment){
  earl.cansCollected = 0;
  earl.row = MAX_DIMENSION-1;
  earl.column = MAX_DIMENSION-1;
  earl.totalReward = 0;
  earl.iterations = 0;
  environment.grid[earl.row][earl.column] = JUST_EARL;
}
// Checks to see if Robby the robot is capturable. If he is capturable, then Earl will capture him for a 
// massive reward.
int performCapture(robot & earl, roboGrid & environment, int action, bool & captured, robot & robby){
  int rowCoord = 0;
  int columnCoord = 1;
  int NWDiag[2];
  int NEDiag[2];
  int SWDiag[2];
  int SEDiag[2];
  int reward = FAILED_CAPTURE_REWARD;
    // First perform a series of simulated coordinate computations in each diagonal direction
    SWDiag[rowCoord] = earl.row-1;
    SWDiag[columnCoord] = earl.column-1;
    
    SEDiag[rowCoord] = earl.row-1;
    SEDiag[columnCoord] = earl.column+1;

    NWDiag[rowCoord] = earl.row+1;
    NWDiag[columnCoord] = earl.column-1;

    NEDiag[rowCoord] = earl.row+1;
    NEDiag[columnCoord] = earl.column+1;

  if(SWDiag[rowCoord] == robby.row && SWDiag[columnCoord] == robby.column){
    captured = true;
  }
  if(SEDiag[rowCoord] == robby.row && SEDiag[columnCoord] == robby.column){
    captured = true;
  }
  if(NWDiag[rowCoord] == robby.row && NWDiag[columnCoord] == robby.column){
    captured = true;
  }
  if(NEDiag[rowCoord] == robby.row && NEDiag[columnCoord] == robby.column){
    captured = true;
  }

  if(captured == true){
    reward = CAPTURE_REWARD;
    cout << "Capture success!" << endl;
  }
  else{
  }
  return reward;
}
// returns updated q-value for weight updating in the q-matrix. Also calls: retrieveNextQval (see robby.cpp)
int q_update(int & currIndex, int action, int reward, qmatrix & nextState, qmatrix ** actiongrid, int lookup){
  int qvalue = 0;
  int maxQ = 0;
  int reply = 0;

  if(lookup == -1){
    qvalue = actiongrid[currIndex]->weights[action] + LEARNING_RATE * (reward + (DISCOUNT_FACTOR * 0) - 
    actiongrid[currIndex]->weights[action]);
    currIndex = addState(nextState, actiongrid); // add the state to this robot's q-table, and also modify the index for the calling method.
    // We add it here because it doesn't exist yet. This step is crucial.
  }
  else{ 
    maxQ = retrieveNextQval(nextState); // Retrieve the maximum perceived q-value achievable in the next state.
    qvalue = actiongrid[currIndex]->weights[action] + LEARNING_RATE * (reward + (DISCOUNT_FACTOR*maxQ)-
    actiongrid[currIndex]->weights[action]);
  }
  return qvalue;
}
// I want to make a preliminary observation of the state I've moved to. If I can see Robby, I want to make that desirable.
int scoutForRobby(robot & earl, roboGrid & environment, robot & robby){
  int rowCoord = 0;
  int colCoord = 1;
  int north[2];
  int east[2];
  int south[2];
  int west[2];
  int reward = 0;
  north[rowCoord] = earl.row+1;
  north[colCoord] = earl.column; // Calculate the coordinates of the tile above earl.
  
  east[rowCoord] = earl.row;
  east[colCoord] = earl.column+1; // calculate the coordinates of the tile to the east of earl.

  south[rowCoord] = earl.row-1; 
  south[colCoord] = earl.column; // same as above, except for south.

  west[rowCoord] = earl.row;
  west[colCoord] = earl.column-1; // same as above, except for west.
  
  if(north[rowCoord] == robby.row && north[colCoord] == robby.column){
    reward = 5;
  }
  if(east[rowCoord] == robby.row && east[colCoord] == robby.column){
    reward = 5;
  }
  if(south[rowCoord] == robby.row && south[colCoord] == robby.column){
    reward = 5;
  }
  if(west[rowCoord] == robby.row && south[colCoord] == robby.column){
    reward = 5;
  }
  if(reward == 5){
    prettyPrintArena(environment, earl, robby);
    cout << " --------------------------------- " << endl;
  }
  return reward;
}
void manageEnvironment(roboGrid & environment, int prevRow, int prevCol, int dirMoved, int flag, robot & testBot){
  int currRow = testBot.row;
  int currCol = testBot.column;
  fourCaseCheck(environment, prevRow, prevCol, flag); // This simulates what the model will be after a character moves off of a tile
  newStateFourCaseCheck(environment, currRow, currCol, flag); // This simulates what the model will be after a character moves ON to a tile.
}
void newStateFourCaseCheck(roboGrid & environment, int row, int col, int flag){
  if(flag == 0){ // if the flag is 0, then I'm cleaning up after Robby's movement.
    if(environment.grid[row][col] == EMPTY){
      environment.grid[row][col] = JUST_ROBBY;
    }
    if(environment.grid[row][col] == CAN){
      environment.grid[row][col] = CAN_ROBBY;
    }
    if(environment.grid[row][col] == JUST_EARL){
      environment.grid[row][col] = JUST_EARL_ROBBY;
    }
    if(environment.grid[row][col] == CAN_EARL){
      environment.grid[row][col] = EARL_ROBBY_CAN;
    }
  }
  else{ // otherwise, I'm cleaning up after Earl's movement.
    if(environment.grid[row][col] == EMPTY){
      environment.grid[row][col] = JUST_EARL;
    }
    if(environment.grid[row][col] == CAN){
      environment.grid[row][col] = CAN_EARL;
    }
    if(environment.grid[row][col] == JUST_ROBBY){
      environment.grid[row][col] = JUST_EARL_ROBBY;
    }
    if(environment.grid[row][col] == CAN_ROBBY){
      environment.grid[row][col] = EARL_ROBBY_CAN;
    }
  }
}
void fourCaseCheck(roboGrid & environment, int prevRow, int prevCol, int flag){
  if(flag == 0){ // if the flag is 0, then I'm cleaning up after Robby's movement.
    if(environment.grid[prevRow][prevCol] == JUST_ROBBY){
      environment.grid[prevRow][prevCol] = EMPTY;
    }
    if(environment.grid[prevRow][prevCol] == CAN_ROBBY){
      environment.grid[prevRow][prevCol] = CAN;
    }
    if(environment.grid[prevRow][prevCol] == JUST_EARL_ROBBY){
      environment.grid[prevRow][prevCol] = JUST_EARL;
    }
    if(environment.grid[prevRow][prevCol] == EARL_ROBBY_CAN){
      environment.grid[prevRow][prevCol] = CAN_EARL;
    }
  }
  else{ // otherwise, I'm cleaning up after Earl's movement. This is entirely just to maintain the grid model.
    if(environment.grid[prevRow][prevCol] == JUST_EARL){
      environment.grid[prevRow][prevCol] = EMPTY;
    }
    if(environment.grid[prevRow][prevCol] == CAN_EARL){
      environment.grid[prevRow][prevCol] = CAN;
    }
    if(environment.grid[prevRow][prevCol] == JUST_EARL_ROBBY){
      environment.grid[prevRow][prevCol] = JUST_ROBBY;
    }
    if(environment.grid[prevRow][prevCol] == EARL_ROBBY_CAN){
      environment.grid[prevRow][prevCol] = CAN_ROBBY;
    }
  }
}
void prettyPrintArena(roboGrid & arena, robot & earl, robot & robby){
  int i = 0;
  int j = 0;
  cout << "Cans on this grid: " << arena.cansPlaced << "//Robby Step: " << robby.iterations << " //Earl Step: "
  << earl.iterations <<  endl;
  for(i = 0; i < MAX_DIMENSION; ++i){
    for(j = 0; j < MAX_DIMENSION; ++j){
      if(arena.grid[i][j] == CAN_ROBBY){
        cout << "CR" << "  ";
      }
      else if(arena.grid[i][j] == JUST_ROBBY){
        cout << "R" << "  ";
      }
      else if(arena.grid[i][j] == CAN_EARL){
        cout << "CE" << "  ";
      }
      else if(arena.grid[i][j] == JUST_EARL){
        cout << "E" << "  ";
      }
      else if(arena.grid[i][j] == EARL_ROBBY_CAN){
        cout << "ERC" << "  ";
      }
      else if(arena.grid[i][j] == JUST_EARL_ROBBY){
        cout << "ER" << "  ";
      }
      else{
        cout << arena.grid[i][j] << "  ";
      }
    }
    cout << endl;
  }
}
