/********************************************************/
// John Lorenz IV // CS 441 // Programming Homework #3  //
// Reinforced learning program that uses Q-learning to  //
// train Robby the robot to navigate a grid and pick up //
// cans.                                                //
/********************************************************/
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
      case 1:
        displayQmatrix(robbyqtable);
        break;
      case 2:
        cout << "How many episodes do you want to TRAIN Robby for?" << endl;
        cin >> episodes;
        cin.ignore(100,'\n');
        cansCollected = 0;
        cansCollected=train_robby(robby, arena, robbyqtable, episodes) + cansCollected;
        cout << "Robby picked up " << cansCollected << " cans during training." << endl;
        break;
      case 3:
        cout << "How many episodes do you want to TEST Robby for?" << endl;
        cin >> episodes;
        cin.ignore(100, '\n');
        cansCollected = 0;
        cansCollected = test_robby(robby, arena, robbyqtable, episodes) + cansCollected;
        cout << "Robby picked up " << cansCollected << " cans during testing." << endl;
        break;
      case 4:
        cout << "How many episodes to TRAIN for?" << endl;
        cin >> episodes;
        cin.ignore(100, '\n');
        cansCollected = 0;
        cansCollected = train_earl_robby(robby, earl, arena, robbyqtable, earlqtable, episodes) + cansCollected;
        cout << "Robby picked up " << cansCollected << " cans during training." << endl;
        break;
      case 5:
        cout << "How many episodes to TEST for?" << endl;
        cansCollected = 0;
//        cansCollected = test_robby_earl(robby, earl, arena, robbyqtable, earlqtable, episodes) + cansCollected;
        cout << "Robby picked up " << cansCollected << " cans during testing." << endl;
        break;
      case 6:
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
      default:
         userChoice = 0;
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
  cout << "1. Print robbyqtable " << endl;
  cout << "2. Train Robby" << endl;
  cout << "3. Run Tests on Robby" << endl;
  cout << "4. Train Robby/Earl" << endl;
  cout << "5. Test Robby/Earl" << endl;
  cout << "6. Print Earl's Qtable" << endl;
  cout << endl;
  cout << "// Debug Functionality //" << endl;
  cout << "32. Print Arena" << endl;
  cout << "33. Perform single observation" << endl;
  cout << "34. Perform an action" << endl;
  cout << "35. Print Robby Data" << endl;
  cout << "0. Quit" << endl;
};
