/*************************************************************/
// John Lorenz IV // CS 441 // Group Programming Assignment  //
// Reinforced learning program that uses Q-learning to       //
// train Robby the robot to navigate a grid and pick up      //
// cans.                                                     //
/*************************************************************/
//Modifications made for group Assignment -- Group members, Tanner Sundwall and Nelson Romaine
// Added an additional agent, Earl. The robot Earl must learn the most effective way to pursue and capture Robby.
// Earl uses the same Q-Learning algorithm as Robby, but the intent of the program is to focus on the behavior developed
// by two agents with separate goals in mind.
// There is extensive documentation and data numbers included with this source file.
// 
// There are modifiable settings in the file robby.h. Within this file, the discount factor, learning rate, and maximum steps 
// may be changed to impact the result of the algorithm. To see the full extent of modifiable parameters, view robby.h. They are
// self documented #define constants.

#include "robby.h"

int main(){
  int userChoice = -1;
  roboGrid arena;
  robot robby;
  robot earl;
  qmatrix * robbyqtable[MAX_QMATRIX];
  qmatrix * earlqtable[MAX_QMATRIX];
  qmatrix currentState;
  qmatrix observing;
  initEnvironment(robby, arena);
  initializeQmatrix(robbyqtable);
  initializeQmatrix(earlqtable);
  refreshGrid(arena); // init grid
  resetRobby(robby, arena); // init Robby
  initializeEarl(earl, arena); // init earl
  srand(time(0)); // seed time
  int actionDirected = 0;
  int cansCollected = 0;
  int episodes = 1;
  bool captured = false;
  cout << "CURRENT SETTINGS: (Changeable in robby.h)" << endl;
  cout << "Discount Factor: " << DISCOUNT_FACTOR << " || Learning Rate: " << LEARNING_RATE << " || Max Steps: " << MAX_STEPS
  << " || Can Placement: " << CAN_PLACEMENT_PERCENT << "%" << endl;
  cout << "Epsilon rate: " << stage1 << "% -> " << stage2 << "% -> " << stage3 << "% -> " << stage4 << "% -> " << stage5 << "%" << endl;
  while(userChoice != 0){
    menuPrompt();
    cin >> userChoice;
    cin.ignore(100, '\n');
    switch(userChoice){
      case 1: // display Robby's q-table
        displayQmatrix(robbyqtable);
        break;
      case 2: // train robby w/o earl
        cout << "How many episodes do you want to TRAIN Robby for?" << endl;
        cin >> episodes;
        cin.ignore(100,'\n');
        cansCollected = 0;
        cansCollected=train_robby(robby, arena, robbyqtable, episodes) + cansCollected;
        cout << "Robby picked up " << cansCollected << " cans during training." << endl;
        break;
      case 3: // test robby w/o earl
        cout << "How many episodes do you want to TEST Robby for?" << endl;
        cin >> episodes;
        cin.ignore(100, '\n');
        cansCollected = 0;
        cansCollected = test_robby(robby, arena, robbyqtable, episodes) + cansCollected;
        cout << "Robby picked up " << cansCollected << " cans during testing." << endl;
        break;
      case 4: // pretty print
        prettyPrintArena(arena, earl, robby);
        break;
      case 5: // train robby w/ earl
        cout << "How many episodes to run Earl & Robby for?" << endl;
        cin >> episodes;
        cin.ignore(100, '\n');
        cansCollected = 0;
        cansCollected = train_earl_robby(robby, earl, arena, robbyqtable, earlqtable, episodes) + cansCollected;
        cout << "Robby picked up " << cansCollected << " cans during training." << endl;
        break;
      case 6:// display earl's q-table
        displayQmatrix(earlqtable);
        break;
      case 32:
        printarena(arena);
        break;
      case 33:
        observe(robby, arena, observing);
        printState(observing);
        break;
      case 34:
        cout << "Choose an action: 0 = North, 1 = East, 2 = South, 3 = West, 4 = Collect" << endl;
        cin >> actionDirected;
        cin.ignore(100,'\n');
        robby.totalReward = performAction(robby, arena, actionDirected, captured) + robby.totalReward;
        break;
      case 35:
        dumpRobbyData(robby);
        break;
      case 0:
        //quits without snarky message
        break;
      default:
         userChoice = 0;
         cout << "Hey man.. it takes time to check edge cases like bad input. Terminating program." << endl;
         //quits program
         break;
    }
  }
  deallocateQmatrix(robbyqtable);
  deallocateQmatrix(earlqtable);
  return 0;
}
void menuPrompt(){
  cout << "Select an option:" << endl;
  cout << "1. Print robbyqtable. " << endl;
  cout << "2. Train Robby." << endl;
  cout << "3. Run Tests on Robby." << endl;
  cout << endl;
  cout << "// Robby the Robot: Earl Edition//" << endl;
  cout << "4. Pretty Print Arena." << endl;
  cout << "5. Train/Test Robby and Earl." << endl;
  cout << "6. Print Earl's Qtable." << endl;
  cout << endl;
  cout << "// Debug Functionality //" << endl;
  cout << "32. Print Arena" << endl;
  cout << "33. Perform single observation" << endl;
  cout << "34. Perform an action" << endl;
  cout << "35. Print Robby Data" << endl;
  cout << "0. Quit" << endl;
};
