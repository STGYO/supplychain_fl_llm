// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract FederatedUpdateRegistry {
    struct TrainingUpdate {
        address nodeAddress;
        uint256 roundNumber;
        string modelHash;
        uint256 timestamp;
    }

    mapping(address => bool) public registeredNodes;
    mapping(uint256 => TrainingUpdate[]) private updatesByRound;

    event NodeRegistered(address indexed nodeAddress, uint256 timestamp);
    event UpdateSubmitted(address indexed nodeAddress, uint256 indexed roundNumber, string modelHash, uint256 timestamp);

    function registerNode() external {
        require(!registeredNodes[msg.sender], "Node already registered");
        registeredNodes[msg.sender] = true;
        emit NodeRegistered(msg.sender, block.timestamp);
    }

    function submitUpdate(uint256 roundNumber, string calldata modelHash) external {
        require(registeredNodes[msg.sender], "Node not registered");
        require(bytes(modelHash).length > 0, "Model hash cannot be empty");

        TrainingUpdate memory item = TrainingUpdate({
            nodeAddress: msg.sender,
            roundNumber: roundNumber,
            modelHash: modelHash,
            timestamp: block.timestamp
        });

        updatesByRound[roundNumber].push(item);

        emit UpdateSubmitted(msg.sender, roundNumber, modelHash, block.timestamp);
    }

    function getUpdates(uint256 roundNumber) external view returns (TrainingUpdate[] memory) {
        return updatesByRound[roundNumber];
    }
}
