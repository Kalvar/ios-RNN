//
//  ViewController.m
//  ios-RNN
//
//  Created by Kalvar Lin on 2017/7/8.
//  Copyright © 2017年 Kalvar Lin. All rights reserved.
//

#import "ViewController.h"

#import "RNN.h"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    NSArray *features = @[// 0
                          @[@1, @1, @1, @1, @1, @1, @1, @1, @1,
                            @1, @0, @0, @0, @0, @0, @0, @0, @1,
                            @1, @0, @0, @0, @0, @0, @0, @0, @1,
                            @1, @1, @1, @1, @1, @1, @1, @1, @1],
                          
                          // 1
                          @[@0, @0, @0, @0, @0, @0, @0, @0, @0,
                            @0, @0, @0, @0, @0, @0, @0, @0, @0,
                            @0, @0, @0, @0, @0, @0, @0, @0, @0,
                            @1, @1, @1, @1, @1, @1, @1, @1, @1],
                          
                          // 2
                          @[@1, @0, @0, @0, @1, @1, @1, @1, @1,
                            @1, @0, @0, @0, @1, @0, @0, @0, @1,
                            @1, @0, @0, @0, @1, @0, @0, @0, @1,
                            @1, @1, @1, @1, @1, @0, @0, @0, @1],
                          
                          // 3
                          @[@1, @0, @0, @0, @1, @0, @0, @0, @1,
                            @1, @0, @0, @0, @1, @0, @0, @0, @1,
                            @1, @0, @0, @0, @1, @0, @0, @0, @1,
                            @1, @1, @1, @1, @1, @1, @1, @1, @1],
                          
                          // 4
                          @[@1, @1, @1, @1, @1, @0, @0, @0, @0,
                            @0, @0, @0, @0, @1, @0, @0, @0, @0,
                            @0, @0, @0, @0, @1, @0, @0, @0, @0,
                            @1, @1, @1, @1, @1, @1, @1, @1, @1],
                          
                          // 5
                          @[@1, @1, @1, @1, @1, @0, @0, @0, @1,
                            @1, @0, @0, @0, @1, @0, @0, @0, @1,
                            @1, @0, @0, @0, @1, @0, @0, @0, @1,
                            @1, @0, @0, @0, @1, @1, @1, @1, @1],
                          
                          // 6
                          @[@1, @1, @1, @1, @1, @1, @1, @1, @1,
                            @1, @0, @0, @0, @1, @0, @0, @0, @1,
                            @1, @0, @0, @0, @1, @0, @0, @0, @1,
                            @1, @0, @0, @0, @1, @1, @1, @1, @1],
                          
                          // 7
                          @[@1, @0, @0, @0, @0, @0, @0, @0, @0,
                            @1, @0, @0, @0, @0, @0, @0, @0, @0,
                            @1, @0, @0, @0, @0, @0, @0, @0, @0,
                            @1, @1, @1, @1, @1, @1, @1, @1, @1],
                          
                          // 8
                          @[@1, @1, @1, @1, @1, @1, @1, @1, @1,
                            @1, @0, @0, @0, @1, @0, @0, @0, @1,
                            @1, @0, @0, @0, @1, @0, @0, @0, @1,
                            @1, @1, @1, @1, @1, @1, @1, @1, @1],
                          
                          // 9
                          @[@1, @1, @1, @1, @1, @0, @0, @0, @0,
                            @1, @0, @0, @0, @1, @0, @0, @0, @0,
                            @1, @0, @0, @0, @1, @0, @0, @0, @0,
                            @1, @1, @1, @1, @1, @1, @1, @1, @1]
                          ];
    
    __block NSArray *targets  =
    @[// 0
      @[@1, @0, @0, @0, @0, @0, @0, @0, @0, @0],
      // 1
      @[@0, @1, @0, @0, @0, @0, @0, @0, @0, @0],
      // 2
      @[@0, @0, @1, @0, @0, @0, @0, @0, @0, @0],
      // 3
      @[@0, @0, @0, @1, @0, @0, @0, @0, @0, @0],
      // 4
      @[@0, @0, @0, @0, @1, @0, @0, @0, @0, @0],
      // 5
      @[@0, @0, @0, @0, @0, @1, @0, @0, @0, @0],
      // 6
      @[@0, @0, @0, @0, @0, @0, @1, @0, @0, @0],
      // 7
      @[@0, @0, @0, @0, @0, @0, @0, @1, @0, @0],
      // 8
      @[@0, @0, @0, @0, @0, @0, @0, @0, @1, @0],
      // 9
      @[@0, @0, @0, @0, @0, @0, @0, @0, @0, @1],
      ];
    
    __block NSMutableArray <RNNPattern *> *patterns = [[NSMutableArray alloc] init];
    [features enumerateObjectsUsingBlock:^(id  _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
        RNNPattern *pattern = [[RNNPattern alloc] initWithFeatures:obj targets:[targets objectAtIndex:idx]];
        [patterns addObject:pattern];
    }];
    
    RNNFetcher *fetcher = [RNNFetcher sharedFetcher];
    //[fetcher removeForKey:@"save1"];
    
    RNN *nn = [fetcher objectForKey:@"save1"];
    if(nn)
    {
        // 來試試做連續推論 (works)
        NSArray <NSArray <NSNumber *> *> *numbers = @[
                                                      // 5
                                                      @[@1, @1, @1, @1, @1, @0, @0, @0, @1,
                                                        @1, @0, @0, @0, @1, @0, @0, @0, @1,
                                                        @1, @0, @0, @0, @1, @0, @0, @0, @1,
                                                        @1, @0, @0, @0, @1, @1, @1, @1, @1],
                                                      
                                                      // 6
                                                      @[@1, @1, @1, @1, @1, @1, @1, @1, @1,
                                                        @1, @0, @0, @0, @1, @0, @0, @0, @1,
                                                        @1, @0, @0, @0, @1, @0, @0, @0, @1,
                                                        @1, @0, @0, @0, @1, @1, @1, @1, @1],
                                                      
                                                      // 不正常的 7
                                                      @[@1, @1, @0, @0, @0, @0, @0, @0, @1,
                                                        @1, @0, @0, @0, @0, @0, @0, @0, @1,
                                                        @1, @0, @0, @0, @0, @0, @0, @0, @1,
                                                        @1, @1, @1, @1, @1, @1, @1, @1, @1]
                                                      ];
        
        __block NSMutableArray <RNNPattern *> *tests = [[NSMutableArray alloc] init];
        [numbers enumerateObjectsUsingBlock:^(NSArray<NSNumber *> * _Nonnull features, NSUInteger idx, BOOL * _Nonnull stop) {
            RNNPattern *pattern = [[RNNPattern alloc] initWithFeatures:features targets:nil];
            [tests addObject:pattern];
        }];
        
        [nn predicateWithPatterns:tests completion:^(NSArray<RNNSequenceOutput *> *sequenceOutputs) {
            [sequenceOutputs enumerateObjectsUsingBlock:^(RNNSequenceOutput * _Nonnull output, NSUInteger idx, BOOL * _Nonnull stop) {
                NSLog(@"(1) Predicated the %li outputs %@", idx, output.networkOutputs);
            }];
        }];
        
        return;
    }
    
    RNN *rnn = [[RNN alloc] init];
    rnn.maxIteration     = 500;
    rnn.convergenceError = 0.001f;
    rnn.learningRate     = 0.5f;
    rnn.timestepSize     = kRNNFullBPTT;
    
    rnn.randomMax        = 0.25f;
    rnn.randomMin        = -0.25f;
    
    [rnn addPatternsFromArray:patterns];
    
    [rnn createHiddenLayerNetsForCount:18];
    [rnn createOutputLayerNetsForCount:10];
    
    [rnn randomizeWeights];
    [rnn uniformActiviation:RNNNetActivationSigmoid];
    
    RNNOptimization *optimization = [[RNNOptimization alloc] init];
    optimization.method           = RNNOptimizationStandardSGD;
    [rnn uniformOptimization:optimization];
    
    [rnn trainingWithIteration:^(NSInteger iteration, RNN *network) {
        NSLog(@"Iteration %li cost %lf", network.iteration, network.costFunction.mse);
    } completion:^(NSInteger totalIteration, RNN *network) {
        
        [network.fetcher save:network forKey:@"save1"];
        
        [network predicateWithPatterns:patterns completion:^(NSArray<RNNSequenceOutput *> *sequenceOutputs) {
            [sequenceOutputs enumerateObjectsUsingBlock:^(RNNSequenceOutput * _Nonnull output, NSUInteger idx, BOOL * _Nonnull stop) {
                NSLog(@"(2) Predicated the %li outputs %@", idx, output.networkOutputs);
            }];
        }];
    }];
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
